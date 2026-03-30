"""
Async Postgres connection via SQLAlchemy + asyncpg.
Used by FastAPI lifespan to init tables and expose a session factory.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

load_dotenv(Path(__file__).parents[1] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Fallback for local dev without Neon — SQLite via aiosqlite
if not DATABASE_URL:
    db_path = Path(__file__).parents[1] / "dev.db"
    DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"
    IS_SQLITE = True
else:
    IS_SQLITE = False

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    # asyncpg needs connect_args only for SSL (handled in URL for Neon)
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables if they don't exist (called at startup)."""
    from app.models import Base  # noqa: import here to avoid circular imports

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
