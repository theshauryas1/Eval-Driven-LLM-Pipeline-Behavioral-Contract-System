"""
POST /trace - receives LLM pipeline traces and enqueues async evaluation.
POST /trace/demo - runs a built-in demo scenario and returns results immediately.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from collections import defaultdict, deque
from threading import Lock
from time import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.evaluators.engine import EvaluationEngine
from app.models import Contract as DbContract
from app.models import EvalResult as DbEvalResult
from app.models import Trace as DbTrace

router = APIRouter(prefix="/trace", tags=["traces"])

_engine: EvaluationEngine | None = None
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("TRACE_RATE_LIMIT_WINDOW_SECONDS", "60")))
RATE_LIMIT_MAX_REQUESTS = max(1, int(os.getenv("TRACE_RATE_LIMIT_MAX_REQUESTS", "5")))
_request_log: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        _engine = EvaluationEngine()
    return _engine


DEMO_SCENARIOS = {
    "hallucination": {
        "pipeline_id": "public-demo",
        "input_text": "Summarize the provided company support policy.",
        "retrieved_context": (
            "Acme Support Policy: Support is available Monday through Friday from 9 AM to 5 PM IST. "
            "Email support@acme.test for billing issues. Refunds are issued only for duplicate charges."
        ),
        "output": (
            "Acme offers 24/7 phone support and guarantees refunds for all cancellations. "
            "Contact support@acme.test for immediate help."
        ),
        "description": "Intentionally broken output with unsupported claims and a policy violation.",
    },
    "clean": {
        "pipeline_id": "public-demo",
        "input_text": "Summarize the provided company support policy.",
        "retrieved_context": (
            "Acme Support Policy: Support is available Monday through Friday from 9 AM to 5 PM IST. "
            "Email support@acme.test for billing issues. Refunds are issued only for duplicate charges."
        ),
        "output": (
            "Support is available Monday through Friday from 9 AM to 5 PM IST. "
            "Refunds are limited to duplicate charges. [Source: support_policy]"
        ),
        "description": "Grounded output that cites the provided context.",
    },
}


class TraceIn(BaseModel):
    pipeline_id: str
    input_text: str
    retrieved_context: str = ""
    output: str


class TraceOut(BaseModel):
    trace_id: str
    message: str


class DemoIn(BaseModel):
    scenario: str = "hallucination"


class DemoOut(BaseModel):
    scenario: str
    description: str
    trace_id: str
    trace: dict
    eval_results: list[dict]
    summary: dict


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request, response: Response) -> None:
    client_ip = _get_client_ip(request)
    now = time()

    with _rate_limit_lock:
        bucket = _request_log[client_ip]
        while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            response.headers["Retry-After"] = str(retry_after)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for {client_ip}. "
                    f"Max {RATE_LIMIT_MAX_REQUESTS} request(s) per {RATE_LIMIT_WINDOW_SECONDS} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)


@router.post("", response_model=TraceOut, status_code=202)
async def ingest_trace(
    payload: TraceIn,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a raw LLM trace and stores it. Evaluation runs asynchronously
    in the background and the endpoint returns immediately with the trace_id.
    """
    _enforce_rate_limit(request, response)

    trace = DbTrace(
        pipeline_id=payload.pipeline_id,
        input_text=payload.input_text,
        retrieved_context=payload.retrieved_context,
        output=payload.output,
    )
    db.add(trace)
    await db.commit()
    await db.refresh(trace)

    background_tasks.add_task(
        _run_evaluation,
        trace_id=str(trace.id),
        output=payload.output,
        retrieved_context=payload.retrieved_context,
        input_text=payload.input_text,
    )

    return TraceOut(
        trace_id=str(trace.id),
        message="Trace accepted. Evaluation running in background.",
    )


@router.post("/demo", response_model=DemoOut)
async def run_demo(
    payload: DemoIn,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Runs a built-in scenario synchronously and returns the full evaluation result."""
    _enforce_rate_limit(request, response)

    scenario = DEMO_SCENARIOS.get(payload.scenario)
    if scenario is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown demo scenario '{payload.scenario}'. Available: {', '.join(sorted(DEMO_SCENARIOS))}",
        )

    trace = DbTrace(
        pipeline_id=scenario["pipeline_id"],
        input_text=scenario["input_text"],
        retrieved_context=scenario["retrieved_context"],
        output=scenario["output"],
    )
    db.add(trace)
    await db.commit()
    await db.refresh(trace)

    results = await _evaluate_trace(
        output=scenario["output"],
        retrieved_context=scenario["retrieved_context"],
        input_text=scenario["input_text"],
    )
    await _persist_eval_results(db, trace.id, results)
    await db.refresh(trace)

    eval_rows = (
        await db.execute(
            DbEvalResult.__table__.select().where(DbEvalResult.trace_id == trace.id)
        )
    ).mappings().all()
    eval_results = [
        {
            "id": str(row["id"]),
            "trace_id": str(row["trace_id"]),
            "contract_id": row["contract_id"],
            "passed": row["passed"],
            "explanation": row["explanation"],
            "reasoning_trace": row["reasoning_trace"],
            "evaluated_at": row["evaluated_at"].isoformat() if row["evaluated_at"] else None,
        }
        for row in eval_rows
    ]

    return DemoOut(
        scenario=payload.scenario,
        description=scenario["description"],
        trace_id=str(trace.id),
        trace=trace.to_dict(),
        eval_results=eval_results,
        summary={
            "total_contracts": len(eval_results),
            "passed": sum(1 for item in eval_results if item["passed"]),
            "failed": sum(1 for item in eval_results if not item["passed"]),
            "violations": [item["contract_id"] for item in eval_results if not item["passed"]],
        },
    )


async def _run_evaluation(
    trace_id: str,
    output: str,
    retrieved_context: str,
    input_text: str,
) -> None:
    from app.database import AsyncSessionLocal

    results = await _evaluate_trace(
        output=output,
        retrieved_context=retrieved_context,
        input_text=input_text,
    )

    async with AsyncSessionLocal() as session:
        await _persist_eval_results(session, uuid.UUID(trace_id), results)


async def _evaluate_trace(
    output: str,
    retrieved_context: str,
    input_text: str,
):
    engine = get_engine()
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: engine.run(
            output=output,
            retrieved_context=retrieved_context,
            input_text=input_text,
        ),
    )


async def _persist_eval_results(
    db: AsyncSession,
    trace_id: uuid.UUID,
    results: list,
) -> None:
    engine = get_engine()

    for result in results:
        existing = await db.get(DbContract, result.contract_id)
        if existing is None:
            contract_obj = next(
                (contract for contract in engine.contracts if contract.id == result.contract_id),
                None,
            )
            if contract_obj:
                db.add(
                    DbContract(
                        id=contract_obj.id,
                        description=contract_obj.description,
                        type=contract_obj.type,
                        config_json=contract_obj.config,
                    )
                )

    await db.flush()

    for result in results:
        db.add(
            DbEvalResult(
                trace_id=trace_id,
                contract_id=result.contract_id,
                passed=result.passed,
                explanation=result.explanation,
                reasoning_trace=result.reasoning_trace,
            )
        )

    await db.commit()
