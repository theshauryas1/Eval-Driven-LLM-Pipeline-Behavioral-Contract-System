"""
GET /contracts — list all known contracts with latest pass rates.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Contract as DbContract
from app.models import EvalResult as DbEvalResult

router = APIRouter(prefix="/contracts", tags=["contracts"])


@router.get("")
async def list_contracts(db: AsyncSession = Depends(get_db)):
    """Returns all contracts with their overall pass rate."""
    contracts = (await db.execute(select(DbContract))).scalars().all()

    result = []
    for c in contracts:
        # Compute overall pass rate across all eval results
        total_q = select(func.count()).where(DbEvalResult.contract_id == c.id)
        pass_q = select(func.count()).where(
            DbEvalResult.contract_id == c.id, DbEvalResult.passed.is_(True)
        )
        total = (await db.execute(total_q)).scalar_one()
        passed = (await db.execute(pass_q)).scalar_one()

        rate = round(passed / total * 100, 1) if total > 0 else None

        result.append({
            **c.to_dict(),
            "eval_count": total,
            "pass_rate": rate,
        })

    return {"contracts": result}
