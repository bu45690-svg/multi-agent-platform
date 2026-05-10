"""
worker/tools/base.py

Base classes and contracts for all tools in the multi-agent platform.

Every tool:
  - Subclasses BaseTool and implements async _run(inp: ToolInput) -> ToolResult
  - Raises one of the typed exceptions so the Orchestrator can branch on failure
  - Is called only via BaseTool.execute() which handles timing, logging, and DB write

Failure contracts
-----------------
ToolTimeoutError    -> Orchestrator waits 1s, retries with smaller budget
ToolInputError      -> Orchestrator logs and skips (bad caller, not bad tool)
ToolExecutionError  -> Orchestrator retries up to MAX_RETRIES, then marks partial
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from worker.context import SharedContext
    from logger.structured import StructuredLogger


# ---------------------------------------------------------------------------
# Typed exceptions — Orchestrator branches on these, never on message text
# ---------------------------------------------------------------------------

class ToolTimeoutError(Exception):
    """Tool did not respond within its deadline."""
    def __init__(self, tool_name: str, timeout_s: float) -> None:
        self.tool_name = tool_name
        self.timeout_s = timeout_s
        super().__init__(f"{tool_name} timed out after {timeout_s}s")


class ToolInputError(Exception):
    """Caller supplied invalid / missing input. Do not retry."""
    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"{tool_name} bad input: {detail}")


class ToolExecutionError(Exception):
    """Tool ran but produced an error. May retry."""
    def __init__(self, tool_name: str, detail: str, retryable: bool = True) -> None:
        self.tool_name = tool_name
        self.detail = detail
        self.retryable = retryable
        super().__init__(f"{tool_name} execution error: {detail}")


# ---------------------------------------------------------------------------
# ToolInput — wrapper passed to every tool
# ---------------------------------------------------------------------------

@dataclass
class ToolInput:
    """Wrapper passed to every tool call."""
    data: dict[str, Any]
    job_id: str
    agent_id: str
    attempt: int = 1

    def require(self, *keys: str) -> None:
        """Raise ToolInputError if any of `keys` are absent or None."""
        missing = [k for k in keys if k not in self.data or self.data[k] is None]
        if missing:
            raise ToolInputError(
                tool_name="(unknown)",
                detail=f"Missing required fields: {missing}",
            )


# ---------------------------------------------------------------------------
# ToolResult — standardised return value from every tool
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standardised return value from every tool."""
    success: bool
    output: Any
    error: str | None = None
    latency_ms: int = 0
    tool_name: str = ""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> "ToolResult":
        """Re-raise as ToolExecutionError if not successful."""
        if not self.success:
            raise ToolExecutionError(
                tool_name=self.tool_name,
                detail=self.error or "unknown error",
            )
        return self


# ---------------------------------------------------------------------------
# BaseTool — subclass and implement _run()
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Abstract base for all tools.

    Subclass contract
    -----------------
    name        : str   — unique snake_case identifier, e.g. "web_search"
    description : str   — one-sentence description
    timeout_s   : float — hard deadline; execute() wraps _run() in asyncio.wait_for

    Implement
    ---------
    async _validate_input(inp: ToolInput) -> None
        Raise ToolInputError on bad input.

    async _run(inp: ToolInput) -> ToolResult
        Core logic. May raise ToolExecutionError.
    """

    name: str = ""
    description: str = ""
    timeout_s: float = 30.0

    def __init__(
        self,
        logger: "StructuredLogger | None" = None,
        db_session_factory: Any | None = None,
    ) -> None:
        self._log = logger
        self._db_session_factory = db_session_factory

    # ------------------------------------------------------------------
    # Public entry point — agents always call this, never _run directly
    # ------------------------------------------------------------------

    async def execute(
        self,
        inp: ToolInput,
        context: "SharedContext | None" = None,
    ) -> ToolResult:
        """
        Run the tool with timing, logging, and DB persistence.

        Raises
        ------
        ToolInputError      — bad input, do not retry
        ToolTimeoutError    — deadline exceeded, retry with smaller scope
        ToolExecutionError  — tool failure, retry if .retryable
        """
        t0 = time.perf_counter()

        # 1. Validate input
        try:
            await self._validate_input(inp)
        except ToolInputError:
            raise
        except Exception as exc:
            raise ToolInputError(self.name, str(exc)) from exc

        # 2. Log call intent
        if self._log:
            await self._log.tool_call(
                inp.job_id,
                tool_name=self.name,
                tool_input=inp.data,
                attempt=inp.attempt,
            )

        # 3. Run with timeout
        try:
            result: ToolResult = await asyncio.wait_for(
                self._run(inp),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - t0) * 1000)
            err = ToolResult(
                success=False,
                output=None,
                error=f"Timed out after {self.timeout_s}s",
                latency_ms=elapsed,
                tool_name=self.name,
            )
            await self._persist(inp, err)
            if self._log:
                await self._log.tool_result(
                    inp.job_id,
                    tool_name=self.name,
                    latency_ms=elapsed,
                    success=False,
                    error=err.error,
                )
            raise ToolTimeoutError(self.name, self.timeout_s)
        except ToolExecutionError:
            raise
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            err = ToolResult(
                success=False,
                output=None,
                error=str(exc),
                latency_ms=elapsed,
                tool_name=self.name,
            )
            await self._persist(inp, err)
            if self._log:
                await self._log.tool_result(
                    inp.job_id,
                    tool_name=self.name,
                    latency_ms=elapsed,
                    success=False,
                    error=str(exc),
                )
            raise ToolExecutionError(self.name, str(exc)) from exc

        # 4. Stamp result with tool_name and latency
        result.tool_name = self.name
        result.latency_ms = int((time.perf_counter() - t0) * 1000)

        # 5. Persist to tool_calls table
        await self._persist(inp, result)

        # 6. Log result
        if self._log:
            await self._log.tool_result(
                inp.job_id,
                tool_name=self.name,
                latency_ms=result.latency_ms,
                success=result.success,
                output_obj=result.output,
                error=result.error,
            )

        return result

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    async def _validate_input(self, inp: ToolInput) -> None:  # noqa: B027
        """Override to add input validation. Raise ToolInputError on failure."""

    @abstractmethod
    async def _run(self, inp: ToolInput) -> ToolResult:
        """Core tool logic. Raise ToolExecutionError on failure."""

    # ------------------------------------------------------------------
    # DB persistence — fire-and-forget, never raises
    # ------------------------------------------------------------------

    async def _persist(self, inp: ToolInput, result: ToolResult) -> None:
        if self._db_session_factory is None:
            return
        try:
            session = self._db_session_factory()
            from sqlalchemy import text
            await session.execute(
                text(
                    "INSERT INTO tool_calls "
                    "(id, job_id, agent_id, tool_name, input, output, error, latency_ms, attempt) "
                    "VALUES "
                    "(:id, :job_id, :agent_id, :tool_name, :input::jsonb, :output::jsonb, "
                    ":error, :latency_ms, :attempt)"
                ),
                {
                    "id":         result.call_id,
                    "job_id":     inp.job_id,
                    "agent_id":   inp.agent_id,
                    "tool_name":  self.name,
                    "input":      json.dumps(inp.data, default=str),
                    "output":     json.dumps(result.output, default=str),
                    "error":      result.error,
                    "latency_ms": result.latency_ms,
                    "attempt":    inp.attempt,
                },
            )
            await session.commit()
        except Exception:
            pass  # DB failure never silences a tool result

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(obj: Any) -> str:
        raw = json.dumps(obj, sort_keys=True, default=str).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]