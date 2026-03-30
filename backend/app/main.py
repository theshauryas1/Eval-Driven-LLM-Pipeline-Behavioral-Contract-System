"""
FastAPI application entrypoint.

Startup: initialises the database tables and syncs contracts from YAML.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).parent.parent / ".env")

from app.database import init_db, AsyncSessionLocal
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    await init_db()
    await _sync_contracts_to_db()
    yield
    # ── Shutdown ─────────────────────────────────────────────


async def _sync_contracts_to_db() -> None:
    """Mirror YAML contract definitions into the DB (upsert)."""
    from app.evaluators.contract_loader import load_contracts
    from app.models import Contract as DbContract

    contracts = load_contracts()
    async with AsyncSessionLocal() as session:
        for c in contracts:
            existing = await session.get(DbContract, c.id)
            if existing is None:
                session.add(
                    DbContract(
                        id=c.id,
                        description=c.description,
                        type=c.type,
                        config_json=c.config,
                    )
                )
            else:
                existing.description = c.description
                existing.type = c.type
                existing.config_json = c.config
        await session.commit()


app = FastAPI(
    title="LLM Behavioral Contract System",
    description=(
        "Eval-driven pipeline that checks every LLM trace against "
        "behavioral contracts and exposes violations via a REST API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "https://shaurya-beta.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "1.0.0"}
