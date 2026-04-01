"""
GET /results — paginated trace list with eval summaries.
GET /results/{trace_id} — full trace + all eval results.
GET /results/stats — pass rate time series for dashboard charts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import EvalResult as DbEvalResult
from app.models import Trace as DbTrace

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/stats")
async def get_stats(
    contract_id: str = Query(..., description="Contract ID to get stats for"),
    days: int = Query(7, ge=1, le=90, description="Number of days of history"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns daily pass rate time series for a specific contract.
    Used by the React dashboard to render line charts.
    """
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    rows = (
        await db.execute(
            select(DbEvalResult.evaluated_at, DbEvalResult.passed)
            .where(
                DbEvalResult.contract_id == contract_id,
                DbEvalResult.evaluated_at >= since,
            )
            .order_by(DbEvalResult.evaluated_at.asc())
        )
    ).all()

    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for evaluated_at, passed in rows:
        if evaluated_at is None:
            continue
        day = evaluated_at.date().isoformat()
        grouped[day]["total"] += 1
        grouped[day]["passed"] += int(bool(passed))

    series = []
    for day, counts in grouped.items():
        total = counts["total"]
        passed = counts["passed"]
        series.append(
            {
                "date": day,
                "total": total,
                "passed": passed,
                "pass_rate": round(passed / total * 100, 1) if total else None,
            }
        )

    return {"contract_id": contract_id, "days": days, "series": series}


@router.get("/{trace_id}")
async def get_trace_detail(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Full trace detail with all eval results and reasoning traces."""
    import uuid as _uuid

    try:
        uid = _uuid.UUID(trace_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid trace_id format")

    trace = (
        await db.execute(
            select(DbTrace)
            .options(selectinload(DbTrace.eval_results))
            .where(DbTrace.id == uid)
        )
    ).scalar_one_or_none()

    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    return {
        "trace": trace.to_dict(),
        "eval_results": [r.to_dict() for r in trace.eval_results],
        "summary": {
            "total_contracts": len(trace.eval_results),
            "passed": sum(1 for r in trace.eval_results if r.passed),
            "failed": sum(1 for r in trace.eval_results if not r.passed),
        },
    }


@router.get("")
async def list_results(
    pipeline_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Paginated list of traces with pass/fail summary per trace."""
    q = select(DbTrace).options(selectinload(DbTrace.eval_results)).order_by(
        DbTrace.created_at.desc()
    )
    if pipeline_id:
        q = q.where(DbTrace.pipeline_id == pipeline_id)

    traces = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    count_q = select(func.count()).select_from(DbTrace)
    if pipeline_id:
        count_q = count_q.where(DbTrace.pipeline_id == pipeline_id)
    total = (await db.execute(count_q)).scalar_one()

    items = []
    for t in traces:
        evals = t.eval_results
        items.append({
            **t.to_dict(),
            "eval_summary": {
                "total": len(evals),
                "passed": sum(1 for r in evals if r.passed),
                "failed": sum(1 for r in evals if not r.passed),
                "violations": [r.contract_id for r in evals if not r.passed],
            },
        })

    return {"total": total, "limit": limit, "offset": offset, "items": items}
