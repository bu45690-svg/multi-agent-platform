"""
worker/token_budget.py

Token budget tracker for the multi-agent pipeline.

Responsibilities
----------------
1. Count tokens in any string or list-of-messages using tiktoken.
2. Track per-agent consumption against the limits set in RoutingPlan.budgets.
3. Enforce budgets: raise BudgetExceededError before an LLM call that would
   go over limit, so the caller can retry with a shorter prompt or skip.
4. Provide a before/after context manager that measures actual token use and
   records it on SharedContext automatically.

Usage
-----
    from worker.token_budget import BudgetTracker, BudgetExceededError

    tracker = BudgetTracker(context)              # one per job

    # Check before an LLM call
    tracker.check("retrieval_agent", prompt_tokens=512)

    # Count + check + record in one go
    async with tracker.track("retrieval_agent", prompt) as reservation:
        response = await llm(prompt)
        reservation.set_completion(response)      # records actual output tokens
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import tiktoken

if TYPE_CHECKING:
    from worker.context import SharedContext

logger = logging.getLogger(__name__)

# Default encoding — cl100k_base covers GPT-4 / Claude tokenisers closely
_DEFAULT_ENCODING = "cl100k_base"

# Minimum tokens we always reserve for the model's completion output.
# If an agent's remaining budget is below this, we treat it as exhausted.
_MIN_COMPLETION_RESERVE = 64


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """
    Raised when a proposed LLM call would exceed an agent's token budget.

    Attributes
    ----------
    agent_id   : the agent that triggered the check
    limit      : the agent's total token budget
    used       : tokens already consumed by this agent
    requested  : additional tokens the call would need
    """

    def __init__(
        self,
        agent_id: str,
        limit: int,
        used: int,
        requested: int,
    ) -> None:
        self.agent_id = agent_id
        self.limit = limit
        self.used = used
        self.requested = requested
        super().__init__(
            f"Agent {agent_id!r} budget exceeded: "
            f"limit={limit}, used={used}, requested={requested}, "
            f"overage={used + requested - limit}"
        )


class NoBudgetAssignedError(Exception):
    """Raised when an agent tries to use the tracker but has no budget entry."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_encoder(model: str = _DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Return a tiktoken encoder, falling back to cl100k_base on any error."""
    try:
        return tiktoken.get_encoding(model)
    except Exception:
        logger.warning(
            "tiktoken: could not load encoding %r, falling back to cl100k_base", model
        )
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding: str = _DEFAULT_ENCODING) -> int:
    """
    Return the number of tokens in `text` using tiktoken.

    This is a module-level utility so tools and agents can call it without
    needing a BudgetTracker instance.

    Parameters
    ----------
    text     : the string to count
    encoding : tiktoken encoding name (default: cl100k_base)
    """
    if not text:
        return 0
    enc = _get_encoder(encoding)
    return len(enc.encode(text))


def count_message_tokens(
    messages: list[dict[str, str]],
    encoding: str = _DEFAULT_ENCODING,
) -> int:
    """
    Approximate token count for a list of chat messages.

    Mirrors the OpenAI formula (every message costs ~4 overhead tokens, and
    the reply priming costs 2).  Close enough for Claude-format messages too.

    Parameters
    ----------
    messages : list of {"role": ..., "content": ...} dicts
    """
    enc = _get_encoder(encoding)
    total = 2   # reply priming
    for msg in messages:
        total += 4  # per-message overhead
        for value in msg.values():
            total += len(enc.encode(str(value)))
    return total


# ---------------------------------------------------------------------------
# Reservation — returned by BudgetTracker.track()
# ---------------------------------------------------------------------------


@dataclass
class Reservation:
    """
    Holds a reserved (estimated) prompt token count.
    After the LLM responds, call set_completion() so the tracker can record
    the actual total.
    """

    agent_id: str
    prompt_tokens: int
    completion_tokens: int = 0
    _tracker: Any = field(default=None, repr=False)   # BudgetTracker back-ref

    def set_completion(self, text: str, encoding: str = _DEFAULT_ENCODING) -> int:
        """
        Count tokens in the completion text and record them on the tracker.
        Returns the completion token count.
        """
        self.completion_tokens = count_tokens(text, encoding)
        if self._tracker is not None:
            # We already reserved prompt_tokens; now add completion on top.
            self._tracker._add(self.agent_id, self.completion_tokens)
        return self.completion_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """
    Per-job token budget tracker.

    Parameters
    ----------
    context  : the job's SharedContext (provides budgets + token_usage)
    encoding : tiktoken encoding to use for all counts in this job
    """

    def __init__(
        self,
        context: "SharedContext",
        encoding: str = _DEFAULT_ENCODING,
    ) -> None:
        self._ctx = context
        self._enc = _get_encoder(encoding)

    # ------------------------------------------------------------------
    # Low-level read helpers
    # ------------------------------------------------------------------

    def budget_for(self, agent_id: str) -> int:
        """Return the token limit for agent_id, or raise if not assigned."""
        plan = self._ctx.routing_plan
        if plan is None or agent_id not in plan.budgets:
            raise NoBudgetAssignedError(
                f"No budget assigned for agent {agent_id!r}. "
                "Has the Orchestrator set context.routing_plan?"
            )
        return plan.budgets[agent_id]

    def used_by(self, agent_id: str) -> int:
        """Return tokens consumed so far by agent_id (0 if none)."""
        return self._ctx.token_usage.get(agent_id, 0)

    def remaining_for(self, agent_id: str) -> int:
        """Return how many tokens agent_id can still spend."""
        return max(0, self.budget_for(agent_id) - self.used_by(agent_id))

    def is_exhausted(self, agent_id: str) -> bool:
        """True when the agent has fewer tokens left than the minimum reserve."""
        return self.remaining_for(agent_id) < _MIN_COMPLETION_RESERVE

    # ------------------------------------------------------------------
    # Counting helpers (public surface for agents / tools)
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        """Count tokens in `text` using this job's encoding."""
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Approximate token count for a list of chat messages."""
        total = 2
        for msg in messages:
            total += 4
            for value in msg.values():
                total += len(self._enc.encode(str(value)))
        return total

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def check(self, agent_id: str, prompt_tokens: int) -> None:
        """
        Assert that spending `prompt_tokens` now would not exceed the budget.
        Raises BudgetExceededError if it would.

        Call this *before* every LLM invocation so the model is never called
        with a known-bad budget.
        """
        limit = self.budget_for(agent_id)
        used = self.used_by(agent_id)
        if used + prompt_tokens > limit:
            # Log the violation on the context so the orchestrator can act
            self._ctx.flag_violation(
                f"budget_exceeded: agent={agent_id}, "
                f"limit={limit}, used={used}, requested={prompt_tokens}"
            )
            raise BudgetExceededError(
                agent_id=agent_id,
                limit=limit,
                used=used,
                requested=prompt_tokens,
            )

    def check_text(self, agent_id: str, text: str) -> int:
        """
        Count tokens in `text`, run check(), and return the token count.
        Convenience wrapper for the common case of checking a prompt string.
        """
        n = self.count(text)
        self.check(agent_id, n)
        return n

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _add(self, agent_id: str, tokens: int) -> None:
        """Add `tokens` to agent_id's running total on SharedContext."""
        self._ctx.record_tokens(agent_id, tokens)

    def record(self, agent_id: str, prompt_text: str, completion_text: str = "") -> int:
        """
        Count and record tokens for a completed prompt+completion pair.
        Returns the total token count.

        Use this when you already have both sides (e.g. after calling the LLM
        and receiving the full response synchronously).
        """
        total = self.count(prompt_text) + self.count(completion_text)
        self._add(agent_id, total)
        return total

    # ------------------------------------------------------------------
    # Context manager — the preferred high-level API
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def track(self, agent_id: str, prompt: str):
        """
        Async context manager for a single LLM call.

        1. Counts prompt tokens.
        2. Enforces the budget (raises BudgetExceededError if over).
        3. Reserves those tokens immediately.
        4. Yields a Reservation so the caller can record completion tokens
           via reservation.set_completion(response_text).
        5. On exception, the prompt tokens are *not* refunded — a failed
           call still cost something, and the retry manager should decide
           what to do next.

        Example
        -------
            async with tracker.track("synthesis_agent", prompt) as res:
                response = await llm_call(prompt)
                res.set_completion(response)
        """
        prompt_tokens = self.count(prompt)
        self.check(agent_id, prompt_tokens)       # raises if over budget
        self._add(agent_id, prompt_tokens)        # reserve immediately

        reservation = Reservation(
            agent_id=agent_id,
            prompt_tokens=prompt_tokens,
            _tracker=self,
        )
        yield reservation

    # ------------------------------------------------------------------
    # Truncation helper
    # ------------------------------------------------------------------

    def truncate_to_budget(
        self,
        agent_id: str,
        text: str,
        reserve_for_completion: int = _MIN_COMPLETION_RESERVE,
    ) -> str:
        """
        Truncate `text` so that it fits within the agent's remaining budget,
        leaving at least `reserve_for_completion` tokens for the model output.

        Returns the (possibly truncated) string. Agents use this to trim RAG
        context before building the final prompt.
        """
        available = self.remaining_for(agent_id) - reserve_for_completion
        if available <= 0:
            return ""

        tokens = self._enc.encode(text)
        if len(tokens) <= available:
            return text

        truncated = self._enc.decode(tokens[:available])
        logger.debug(
            "truncate_to_budget: agent=%s trimmed %d → %d tokens",
            agent_id,
            len(tokens),
            available,
        )
        return truncated

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict[str, dict[str, int]]:
        """
        Return a summary dict of budget vs usage for every agent in the plan.

            {
              "retrieval_agent": {"budget": 4000, "used": 812, "remaining": 3188},
              ...
            }
        """
        plan = self._ctx.routing_plan
        if plan is None:
            return {}

        result: dict[str, dict[str, int]] = {}
        for agent_id, budget in plan.budgets.items():
            used = self.used_by(agent_id)
            result[agent_id] = {
                "budget":    budget,
                "used":      used,
                "remaining": max(0, budget - used),
            }
        return result

    def __repr__(self) -> str:
        try:
            lines = ", ".join(
                f"{a}={v['used']}/{v['budget']}"
                for a, v in self.report().items()
            )
            return f"BudgetTracker({lines})"
        except Exception:
            return "BudgetTracker(<no plan>"