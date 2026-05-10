"""
logger/structured.py

Structured JSON logger for the multi-agent platform.

Every agent, tool, and orchestrator call emits a log entry through this module.
Entries are written to:
  1. stdout  — newline-delimited JSON, visible in `docker logs`
  2. PostgreSQL `logs` table — queryable for trace reconstruction

Usage:
    from logger.structured import get_logger

    log = get_logger("retrieval_agent")

    async with log.span(job_id, event_type="agent_complete") as span:
        result = await do_work()
        span.set(token_count=512, payload={"chunks_retrieved": 3})

    # Or fire-and-forget:
    await log.emit(job_id, event_type="tool_call", payload={...})
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Internal stdlib logger (only used for bootstrap errors before DB is ready)
# ---------------------------------------------------------------------------
_boot_logger = logging.getLogger("structured_bootstrap")
_boot_logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Valid event_type literals — enforced at emit time to catch typos early
# ---------------------------------------------------------------------------
VALID_EVENT_TYPES = frozenset(
    {
        "agent_start",
        "agent_complete",
        "agent_error",
        "tool_call",
        "tool_result",
        "retry",
        "budget_violation",
        "routing_decision",
        "sse_token",
        "job_start",
        "job_complete",
        "job_error",
    }
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        raw = json.dumps(obj, sort_keys=True, default=str).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return None


def _write_stdout(entry: dict) -> None:
    try:
        sys.stdout.write(json.dumps(entry, default=str) + "\n")
        sys.stdout.flush()
    except Exception as exc:
        _boot_logger.warning("stdout log write failed: %s", exc)


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------

async def _write_db(session: AsyncSession | None, entry: dict) -> None:
    if session is None:
        return
    try:
        from sqlalchemy import text

        await session.execute(
            text("""
                INSERT INTO logs
                    (job_id, agent_id, event_type, latency_ms, token_count,
                     input_hash, output_hash, policy_violations, payload, timestamp)
                VALUES
                    (:job_id, :agent_id, :event_type, :latency_ms, :token_count,
                     :input_hash, :output_hash,
                     :policy_violations,
                     :payload,
                     :timestamp)
            """),
            {
                "job_id": entry.get("job_id"),
                "agent_id": entry.get("agent_id"),
                "event_type": entry.get("event_type"),
                "latency_ms": entry.get("latency_ms"),
                "token_count": entry.get("token_count"),
                "input_hash": entry.get("input_hash"),
                "output_hash": entry.get("output_hash"),

                # ✅ pass Python objects directly
                "policy_violations": entry.get("policy_violations", []),
                "payload": entry.get("payload", {}),

                "timestamp": entry.get("timestamp"),
            },
        )
        await session.commit()

    except Exception as exc:
        _boot_logger.warning("DB log write failed (job=%s): %s", entry.get("job_id"), exc)

    finally:
        # FIX: ensure connection is released
        try:
            if session is not None:
                await session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------

class SpanContext:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, **kwargs: Any) -> None:
        self._data.update(kwargs)

    def set_input(self, obj: Any) -> None:
        self._data["input_hash"] = _sha256(obj)

    def set_output(self, obj: Any) -> None:
        self._data["output_hash"] = _sha256(obj)

    def flag_policy_violation(self, reason: str) -> None:
        violations: list[str] = self._data.setdefault("policy_violations", [])
        violations.append(reason)

    def _snapshot(self) -> dict:
        return dict(self._data)


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class StructuredLogger:
    def __init__(
        self,
        agent_id: str,
        db_session_factory: Any | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._db_session_factory = db_session_factory

    async def emit(
        self,
        job_id: str,
        event_type: str,
        *,
        latency_ms: int | None = None,
        token_count: int | None = None,
        input_obj: Any = None,
        output_obj: Any = None,
        policy_violations: list[str] | None = None,
        payload: dict | None = None,
    ) -> dict:

        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event_type!r}. "
                f"Valid types: {sorted(VALID_EVENT_TYPES)}"
            )

        entry: dict[str, Any] = {
            "timestamp": _now_iso(),
            "job_id": job_id,
            "agent_id": self.agent_id,
            "event_type": event_type,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "input_hash": _sha256(input_obj),
            "output_hash": _sha256(output_obj),
            "policy_violations": policy_violations or [],
            "payload": payload or {},
        }

        _write_stdout(entry)

        session: AsyncSession | None = None
        if self._db_session_factory is not None:
            try:
                session = self._db_session_factory()
            except Exception:
                pass

        # FIX: modern asyncio usage
        asyncio.create_task(self._safe_write(session, entry))

        return entry

    async def _safe_write(self, session: AsyncSession | None, entry: dict) -> None:
        try:
            await _write_db(session, entry)
        except Exception as exc:
            _boot_logger.warning("DB write failed: %s", exc)


    # ------------------------------------------------------------------

    @asynccontextmanager
    async def span(
        self,
        job_id: str,
        event_type: str,
        *,
        payload: dict | None = None,
    ):
        ctx = SpanContext()
        if payload:
            ctx.set(payload=payload)

        t0 = time.perf_counter()
        try:
            yield ctx
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            snap = ctx._snapshot()
            await self.emit(
                job_id,
                "agent_error",
                latency_ms=elapsed,
                policy_violations=snap.get("policy_violations"),
                payload={
                    **(snap.get("payload") or {}),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise
        else:
            elapsed = int((time.perf_counter() - t0) * 1000)
            snap = ctx._snapshot()

            # FIX: avoid double hashing bug
            await self.emit(
                job_id,
                event_type,
                latency_ms=elapsed,
                token_count=snap.get("token_count"),
                input_obj=None,
                output_obj=None,
                policy_violations=snap.get("policy_violations"),
                payload=snap.get("payload"),
            )

    # ------------------------------------------------------------------

    async def agent_start(self, job_id: str, *, input_obj: Any = None) -> dict:
        return await self.emit(job_id, "agent_start", input_obj=input_obj)

    async def agent_complete(
        self,
        job_id: str,
        *,
        latency_ms: int,
        token_count: int | None = None,
        output_obj: Any = None,
        payload: dict | None = None,
    ) -> dict:
        return await self.emit(
            job_id,
            "agent_complete",
            latency_ms=latency_ms,
            token_count=token_count,
            output_obj=output_obj,
            payload=payload,
        )

    async def tool_call(
        self,
        job_id: str,
        *,
        tool_name: str,
        tool_input: Any,
        attempt: int = 1,
    ) -> dict:
        return await self.emit(
            job_id,
            "tool_call",
            input_obj=tool_input,
            payload={"tool_name": tool_name, "attempt": attempt},
        )

    async def tool_result(
        self,
        job_id: str,
        *,
        tool_name: str,
        latency_ms: int,
        success: bool,
        output_obj: Any = None,
        error: str | None = None,
    ) -> dict:
        return await self.emit(
            job_id,
            "tool_result",
            latency_ms=latency_ms,
            output_obj=output_obj,
            payload={"tool_name": tool_name, "success": success, "error": error},
        )

    async def retry(self, job_id: str, *, attempt: int, reason: str, next_budget: int | None = None) -> dict:
        return await self.emit(
            job_id,
            "retry",
            payload={"attempt": attempt, "reason": reason, "next_budget": next_budget},
        )

    async def budget_violation(self, job_id: str, *, limit: int, used: int) -> dict:
        return await self.emit(
            job_id,
            "budget_violation",
            payload={"limit": limit, "used": used, "overage": used - limit},
        )

    async def routing_decision(
        self,
        job_id: str,
        *,
        agents: list[str],
        budgets: dict[str, int],
        reasoning: str,
    ) -> dict:
        return await self.emit(
            job_id,
            "routing_decision",
            payload={"agents": agents, "budgets": budgets, "reasoning": reasoning},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_db_session_factory: Any | None = None


def configure(db_session_factory: Any) -> None:
    global _db_session_factory
    _db_session_factory = db_session_factory


def get_logger(agent_id: str) -> StructuredLogger:
    return StructuredLogger(agent_id=agent_id, db_session_factory=_db_session_factory)
