"""
GET /contracts — list all known contracts with latest pass rates.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Contract as DbContract
from app.models import EvalResult as DbEvalResult

router = APIRouter(prefix="/contracts", tags=["contracts"])


@router.get("")
async def list_contracts(db: AsyncSession = Depends(get_db)):
    """Returns all contracts with their overall pass rate."""
    contracts = (await db.execute(select(DbContract))).scalars().all()
    stats_rows = (
        await db.execute(
            select(
                DbEvalResult.contract_id,
                func.count().label("total"),
                func.sum(DbEvalResult.passed.cast(Integer)).label("passed"),
            )
            .group_by(DbEvalResult.contract_id)
        )
    ).all()

    stats_by_contract = {
        contract_id: {"total": total, "passed": passed or 0}
        for contract_id, total, passed in stats_rows
    }

    result = []
    for c in contracts:
        stats = stats_by_contract.get(c.id, {"total": 0, "passed": 0})
        total = stats["total"]
        passed = stats["passed"]

        rate = round(passed / total * 100, 1) if total > 0 else None

        result.append({
            **c.to_dict(),
            "eval_count": total,
            "pass_rate": rate,
        })

    return {"contracts": result}
