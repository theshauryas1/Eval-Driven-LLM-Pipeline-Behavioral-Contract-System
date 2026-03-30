"""
POST /trace — receives LLM pipeline traces and enqueues async evaluation.
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Contract as DbContract
from app.models import EvalResult as DbEvalResult
from app.models import Trace as DbTrace
from app.evaluators.engine import EvaluationEngine

router = APIRouter(prefix="/trace", tags=["traces"])

# Shared engine instance — contracts loaded once at startup
_engine: EvaluationEngine | None = None


def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        _engine = EvaluationEngine()
    return _engine


class TraceIn(BaseModel):
    pipeline_id: str
    input_text: str
    retrieved_context: str = ""
    output: str


class TraceOut(BaseModel):
    trace_id: str
    message: str


@router.post("", response_model=TraceOut, status_code=202)
async def ingest_trace(
    payload: TraceIn,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a raw LLM trace and stores it. Evaluation runs asynchronously
    in the background — the endpoint returns immediately with the trace_id.
    """
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


async def _run_evaluation(
    trace_id: str,
    output: str,
    retrieved_context: str,
    input_text: str,
) -> None:
    """Background task: runs all evaluators and stores results."""
    from app.database import AsyncSessionLocal

    engine = get_engine()
    results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: engine.run(
            output=output,
            retrieved_context=retrieved_context,
            input_text=input_text,
        ),
    )

    async with AsyncSessionLocal() as session:
        # Upsert contracts referenced by evaluation
        for r in results:
            existing = await session.get(DbContract, r.contract_id)
            if existing is None:
                contract_obj = next(
                    (c for c in engine.contracts if c.id == r.contract_id), None
                )
                if contract_obj:
                    session.add(
                        DbContract(
                            id=contract_obj.id,
                            description=contract_obj.description,
                            type=contract_obj.type,
                            config_json=contract_obj.config,
                        )
                    )

        await session.flush()

        for r in results:
            session.add(
                DbEvalResult(
                    trace_id=uuid.UUID(trace_id),
                    contract_id=r.contract_id,
                    passed=r.passed,
                    explanation=r.explanation,
                    reasoning_trace=r.reasoning_trace,
                )
            )

        await session.commit()
