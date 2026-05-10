"""
worker/context.py

SharedContext — the single Pydantic object that flows through every agent.

Design rules:
  • Agents NEVER call each other. They only read from / write to this object.
  • Every field is typed and validated by Pydantic so bad data fails loudly
    at assignment, not silently at read time two agents later.
  • `sse_queue` (asyncio.Queue) is excluded from serialisation — it is a
    live runtime handle, not data.
  • Helper methods live here so agents stay thin: context.add_chunk(...),
    context.add_claim(...), context.flag_violation(...).

Serialisation
  • context.to_snapshot() → dict  (safe for JSON / PostgreSQL JSONB storage)
  • SharedContext.from_snapshot(d) → SharedContext  (round-trip restore)
    The sse_queue is NOT restored; callers must re-attach it after loading.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models (each maps to a typed list inside SharedContext)
# ---------------------------------------------------------------------------


class TaskNode(BaseModel):
    """One node in the decomposition DAG produced by DecompositionAgent."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    task_type: Literal["retrieve", "compute", "synthesize"]
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    output: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("dependencies")
    @classmethod
    def _no_self_dep(cls, deps: list[str], info: Any) -> list[str]:
        task_id = (info.data or {}).get("task_id")
        if task_id and task_id in deps:
            raise ValueError(f"TaskNode {task_id!r} cannot depend on itself.")
        return deps

    # Convenience mutators (return self for chaining)

    def mark_running(self) -> "TaskNode":
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)
        return self

    def mark_done(self, output: dict[str, Any] | None = None) -> "TaskNode":
        self.status = "done"
        self.output = output
        self.finished_at = datetime.now(timezone.utc)
        return self

    def mark_failed(self, reason: str) -> "TaskNode":
        self.status = "failed"
        self.output = {"error": reason}
        self.finished_at = datetime.now(timezone.utc)
        return self


class RetrievedChunk(BaseModel):
    """A single passage returned by the RAG retriever."""

    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    source_file: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieval_method: Literal["faiss", "web_search", "sql"] = "faiss"

    @field_validator("content")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content must not be empty.")
        return v


class Claim(BaseModel):
    """
    An atomic factual assertion extracted by RetrievalAgent and later
    scored by CritiqueAgent.
    """

    claim_id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:8]}")
    text: str
    source_agent: str
    source_chunk_id: str | None = None   # None → agent-generated, not grounded
    confidence: float = Field(ge=0.0, le=1.0)
    flagged: bool = False
    flag_reason: str | None = None

    def flag(self, reason: str) -> "Claim":
        self.flagged = True
        self.flag_reason = reason
        return self


class CritiqueResult(BaseModel):
    """CritiqueAgent's verdict on a single Claim."""

    claim_id: str
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool = False
    reason: str | None = None
    contradicts: list[str] = Field(default_factory=list)   # other claim_ids


class ProvenanceEntry(BaseModel):
    """Maps a sentence in the final answer back to its source."""

    sentence: str
    source_agent: str
    source_chunk_id: str | None = None


class RoutingPlan(BaseModel):
    """
    The Orchestrator's decision: which agents to run and how many tokens each
    may spend.
    """

    agents: list[str]                      # ordered list of agent_ids
    budgets: dict[str, int]                # agent_id → max tokens
    reasoning: str

    @model_validator(mode="after")
    def _budgets_cover_agents(self) -> "RoutingPlan":
        missing = [a for a in self.agents if a not in self.budgets]
        if missing:
            raise ValueError(
                f"RoutingPlan.budgets missing entries for agents: {missing}"
            )
        return self


# ---------------------------------------------------------------------------
# SharedContext
# ---------------------------------------------------------------------------


class SharedContext(BaseModel):
    """
    The single shared object that flows through the entire agent pipeline.

    Lifecycle
    ---------
    1. Orchestrator creates it:  ctx = SharedContext(job_id=..., query=...)
    2. Each agent reads + writes relevant fields.
    3. After the pipeline, ctx.to_snapshot() is stored in PostgreSQL.
    4. SSE events are pushed via ctx.push_sse(token).

    Thread / task safety
    --------------------
    All agents run sequentially inside the async worker (no concurrency between
    agents for the same job), so plain attribute mutation is safe. If you later
    parallelise agents, wrap mutations in an asyncio.Lock.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Orchestrator outputs ───────────────────────────────────────────────
    routing_plan: RoutingPlan | None = None

    # ── DecompositionAgent outputs ─────────────────────────────────────────
    task_dag: list[TaskNode] = Field(default_factory=list)

    # ── RetrievalAgent outputs ─────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)

    # ── CritiqueAgent outputs ──────────────────────────────────────────────
    critique_results: dict[str, CritiqueResult] = Field(default_factory=dict)
    # key = claim_id

    # ── SynthesisAgent outputs ─────────────────────────────────────────────
    final_answer: str | None = None
    provenance_map: list[ProvenanceEntry] = Field(default_factory=list)

    # ── Cross-cutting concerns ─────────────────────────────────────────────
    token_usage: dict[str, int] = Field(default_factory=dict)
    # key = agent_id, value = tokens consumed

    policy_violations: list[str] = Field(default_factory=list)

    # ── Runtime handle (excluded from serialisation) ───────────────────────
    sse_queue: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: RetrievedChunk) -> RetrievedChunk:
        self.retrieved_chunks.append(chunk)
        return chunk

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None:
        return next((c for c in self.retrieved_chunks if c.chunk_id == chunk_id), None)

    # ------------------------------------------------------------------
    # Claim helpers
    # ------------------------------------------------------------------

    def add_claim(self, claim: Claim) -> Claim:
        self.claims.append(claim)
        return claim

    def get_claim(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)

    @property
    def flagged_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.flagged]

    # ------------------------------------------------------------------
    # Task DAG helpers
    # ------------------------------------------------------------------

    def add_task(self, node: TaskNode) -> TaskNode:
        self.task_dag.append(node)
        return node

    def get_task(self, task_id: str) -> TaskNode | None:
        return next((t for t in self.task_dag if t.task_id == task_id), None)

    def ready_tasks(self) -> list[TaskNode]:
        """
        Return tasks whose dependencies are all 'done' and which are
        themselves still 'pending'.
        """
        done_ids = {t.task_id for t in self.task_dag if t.status == "done"}
        return [
            t
            for t in self.task_dag
            if t.status == "pending" and set(t.dependencies).issubset(done_ids)
        ]

    # ------------------------------------------------------------------
    # Critique helpers
    # ------------------------------------------------------------------

    def set_critique(self, result: CritiqueResult) -> None:
        self.critique_results[result.claim_id] = result
        # Propagate the flagged flag back onto the source Claim object
        claim = self.get_claim(result.claim_id)
        if claim and result.flagged:
            claim.flag(result.reason or "flagged by critique")

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def record_tokens(self, agent_id: str, count: int) -> None:
        self.token_usage[agent_id] = self.token_usage.get(agent_id, 0) + count

    @property
    def total_tokens(self) -> int:
        return sum(self.token_usage.values())

    # ------------------------------------------------------------------
    # Policy violations
    # ------------------------------------------------------------------

    def flag_violation(self, reason: str) -> None:
        self.policy_violations.append(reason)

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------

    async def push_sse(self, token: str) -> None:
        """
        Push a streaming token to the SSE queue.
        No-ops safely if the queue was never attached (e.g. in tests).
        """
        if self.sse_queue is not None:
            await self.sse_queue.put(token)

    async def close_sse(self) -> None:
        """Signal the SSE stream that the job is complete."""
        if self.sse_queue is not None:
            await self.sse_queue.put(None)   # sentinel

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        """
        Return a JSON-safe dict suitable for PostgreSQL JSONB storage.
        `sse_queue` is excluded automatically by `exclude=True` on the field.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "SharedContext":
        """
        Reconstruct a SharedContext from a stored snapshot.
        Callers must re-attach `sse_queue` themselves if streaming is needed.
        """
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Debug repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"SharedContext("
            f"job={self.job_id[:8]}…, "
            f"tasks={len(self.task_dag)}, "
            f"chunks={len(self.retrieved_chunks)}, "
            f"claims={len(self.claims)}, "
            f"flagged={len(self.flagged_claims)}, "
            f"tokens={self.total_tokens}, "
            f"violations={len(self.policy_violations)})"
        )