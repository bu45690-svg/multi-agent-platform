"""
worker/agents/base.py

BaseAgent — abstract base class for every agent in the pipeline.

All agents share:
  • A structured logger scoped to their agent_id
  • Budget-aware LLM calls via BudgetTracker
  • Prompt loading from the database (with in-memory fallback)
  • A standard run() lifecycle: validate → execute → record
  • SSE token streaming helpers
  • A policy-check hook subclasses can override

LLM backend: Google Gemini (gemini-2.5-flash)
The Gemini SDK is synchronous — every call is wrapped in asyncio.to_thread()
so it never blocks the FastAPI / asyncio event loop.

Subclass contract
-----------------
Implement:
    async _execute(context: SharedContext) → None

Optionally override:
    async _validate(context: SharedContext) → None
    def _default_prompt() → str
    def _policy_check(output: str) → list[str]
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from logger.structured import get_logger
from worker.token_budget import BudgetTracker, BudgetExceededError, NoBudgetAssignedError

if TYPE_CHECKING:
    from worker.context import SharedContext


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Parameters
    ----------
    agent_id           : unique snake_case name — set as class attribute in subclasses
    db_session_factory : zero-arg async callable -> AsyncSession (or None)
    """

    agent_id: str = ""
    model: str = "gemini-2.5-flash"
    max_tokens: int = 1024

    def __init__(self, db_session_factory=None) -> None:
        if not self.agent_id:
            raise ValueError(
                f"{type(self).__name__} must set agent_id as a class attribute"
            )
        self._db = db_session_factory
        self._log = get_logger(self.agent_id)

        # Gemini client — one instance shared across all calls in this agent
        from google import genai
        self._gemini = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Public entry point — Orchestrator always calls this, never _execute
    # ------------------------------------------------------------------

    async def run(self, context: "SharedContext") -> None:
        """
        Full agent lifecycle:
          1. Log agent_start
          2. Run _validate()
          3. Run _execute()
          4. Log agent_complete or agent_error
        """
        t0 = time.perf_counter()
        await self._log.agent_start(context.job_id)

        try:
            await self._validate(context)
            await self._execute(context)
        except (BudgetExceededError, NoBudgetAssignedError):
            raise   # Orchestrator handles budget errors directly
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            await self._log.emit(
                context.job_id,
                "agent_error",
                latency_ms=elapsed,
                payload={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise

        elapsed = int((time.perf_counter() - t0) * 1000)
        await self._log.agent_complete(
            context.job_id,
            latency_ms=elapsed,
            token_count=context.token_usage.get(self.agent_id, 0),
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, context: "SharedContext") -> None:
        """Core agent logic. Write all outputs to context."""

    async def _validate(self, context: "SharedContext") -> None:  # noqa: B027
        """Pre-flight checks. Raise ValueError with a clear message on failure."""

    def _default_prompt(self) -> str:
        """Fallback system prompt when the DB has no active entry for this agent."""
        return f"You are the {self.agent_id}. Follow instructions precisely."

    def _policy_check(self, output: str) -> list[str]:
        """
        Scan model output for policy violations.
        Returns a list of violation strings — empty list means clean.
        Override in subclasses for agent-specific checks.
        """
        violations: list[str] = []
        lower = output.lower()
        for phrase in [
            "ignore previous instructions",
            "ignore all instructions",
            "jailbreak",
        ]:
            if phrase in lower:
                violations.append(f"policy: forbidden phrase detected: {phrase!r}")
        return violations

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    async def _load_prompt(self, version: int | None = None) -> str:
        """
        Load the active system prompt for this agent from the DB.
        Falls back to _default_prompt() if DB is unavailable or empty.
        """
        if self._db is None:
            return self._default_prompt()

        try:
            from sqlalchemy import text
            session = self._db()
            if version is not None:
                row = await session.execute(
                    text(
                        "SELECT content FROM prompts "
                        "WHERE agent_id = :aid AND version = :ver LIMIT 1"
                    ),
                    {"aid": self.agent_id, "ver": version},
                )
            else:
                row = await session.execute(
                    text(
                        "SELECT content FROM prompts "
                        "WHERE agent_id = :aid AND is_active = TRUE "
                        "ORDER BY version DESC LIMIT 1"
                    ),
                    {"aid": self.agent_id},
                )
            result = row.fetchone()
            return result[0] if result else self._default_prompt()
        except Exception:
            return self._default_prompt()

    # ------------------------------------------------------------------
    # Core LLM call — Gemini, budget-aware, non-blocking
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        context: "SharedContext",
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """
        Call Gemini with budget enforcement.

        Flow
        ----
        1. Build flat prompt text for token counting.
        2. Open tracker.track() — checks + reserves prompt tokens.
           Raises BudgetExceededError if over limit.
        3. Call Gemini via asyncio.to_thread() — SDK is synchronous.
        4. Record actual completion tokens via reservation.set_completion().
        5. Simulate SSE streaming (word-by-word) if stream=True.
        6. Run policy check; flag violations on context.

        Returns the full completion text string.
        """
        tracker = BudgetTracker(context)

        # Flat text for tiktoken counting
        prompt_text = " ".join(
            [system or ""] + [m.get("content", "") for m in messages]
        )

        # Build Gemini contents list
        # System instruction goes first as plain text.
        # Role prefixes help Gemini understand multi-turn structure.
        contents: list[str] = []
        if system:
            contents.append(f"[SYSTEM INSTRUCTIONS]\n{system}\n[END SYSTEM INSTRUCTIONS]")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                contents.append(f"Assistant: {content}")
            else:
                contents.append(f"User: {content}")

        async with tracker.track(self.agent_id, prompt_text) as reservation:

            # -----------------------------------------------------------
            # Gemini SDK is synchronous — wrap in thread to avoid
            # blocking the asyncio event loop
            # -----------------------------------------------------------
            def _sync_call() -> str:
                response = self._gemini.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                return response.text or ""

            try:
                full_text = await asyncio.to_thread(_sync_call)
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini API call failed [{self.agent_id}]: {exc}"
                ) from exc

            reservation.set_completion(full_text)

        # -------------------------------------------------------------------
        # Simulate streaming via SSE queue
        # Gemini sync SDK returns the full response at once; we push it
        # word-by-word so the SSE client sees incremental output.
        # -------------------------------------------------------------------
        if stream and full_text:
            for word in full_text.split(" "):
                await context.push_sse(word + " ")

        # -------------------------------------------------------------------
        # Policy check — flag violations but do NOT suppress the response
        # -------------------------------------------------------------------
        for violation in self._policy_check(full_text):
            context.flag_violation(violation)
            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={"policy_violation": violation},
            )

        return full_text

    # ------------------------------------------------------------------
    # JSON parsing helper — shared across all agents
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Any:
        """
        Extract and parse the first JSON object or array from `text`.
        Strips markdown fences automatically.
        Raises ValueError on parse failure.
        """
        import json
        import re

        # Strip markdown fences if the model added them despite instructions
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)

        # Find outermost { } or [ ]
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start: end + 1])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"No valid JSON found in LLM output: {text[:200]!r}")