"""
db/session.py

Async SQLAlchemy engine + session factory.

All database access throughout the application uses the async session.
The synchronous engine is kept for Alembic migrations only (Alembic
does not support async natively without adapters, and we're using raw
SQL migrations anyway — it's just here for completeness).

Usage
-----
    from db.session import get_session

    async with get_session() as session:
        result = await session.execute(select(Prompt))
        prompts = result.scalars().all()

In FastAPI, use the `get_db` dependency from api/main.py instead.

Environment variables
---------------------
POSTGRES_USER      default: agent
POSTGRES_PASSWORD  default: secret
POSTGRES_HOST      default: localhost
POSTGRES_PORT      default: 5432
POSTGRES_DB        default: agentdb

Alternatively, set DATABASE_URL directly as a full connection string:
    postgresql://user:password@host:port/dbname
The function will auto-convert it to the asyncpg driver format.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


# ---------------------------------------------------------------------------
# Database URL construction
# ---------------------------------------------------------------------------

def _build_database_url() -> str:
    """
    Build the asyncpg-compatible database URL from environment variables.

    Priority:
      1. DATABASE_URL env var (auto-converted to asyncpg driver)
      2. Individual POSTGRES_* env vars
      3. Hardcoded defaults for local development
    """
    raw = os.environ.get("DATABASE_URL", "")
    if raw:
        # Convert postgres:// or postgresql:// → postgresql+asyncpg://
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        # Already asyncpg?
        if "asyncpg" not in raw:
            raw = raw.replace("postgresql", "postgresql+asyncpg", 1)
        return raw

    user     = os.environ.get("POSTGRES_USER",     "agent")
    password = os.environ.get("POSTGRES_PASSWORD", "secret")
    host     = os.environ.get("POSTGRES_HOST",     "localhost")
    port     = os.environ.get("POSTGRES_PORT",     "5432")
    db       = os.environ.get("POSTGRES_DB",       "agentdb")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


DATABASE_URL: str = _build_database_url()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    DATABASE_URL,
    # Pool settings appropriate for a take-home scale service.
    # Increase pool_size / max_overflow for production workloads.
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,   # verify connections are alive before handing them out
    echo=False,           # set to True to log all SQL for debugging
)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # keep ORM attributes accessible after commit
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.

    Behaviour:
      • Commits automatically on clean exit.
      • Rolls back on any exception before re-raising.
      • Always closes the session in the finally block.

    Example
    -------
        async with get_session() as db:
            db.add(SomeModel(...))
            # commit happens automatically on __aexit__

        # For read-only queries, commit is a no-op — no overhead.
        async with get_session() as db:
            rows = await db.execute(select(Log).where(Log.job_id == job_id))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Dependency for FastAPI (used in api/main.py via Depends)
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a session per request.

    Usage in route handlers:
        @app.get("/something")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Engine disposal (called on application shutdown)
# ---------------------------------------------------------------------------

async def close_engine() -> None:
    """
    Dispose of all pooled connections.
    Call this in the FastAPI lifespan shutdown handler.
    """
    await engine.dispose()