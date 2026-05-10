"""
db/models.py
SQLAlchemy ORM models for the multi-agent orchestration platform.

Design notes:
- All UUIDs generated server-side by Postgres (server_default=text("gen_random_uuid()"))
  so that bulk inserts don't require a Python UUID call per row.
- JSONB columns typed as JSON on the Python side; SQLAlchemy handles serialisation.
- Enums are plain TEXT with CHECK constraints in SQL — no SQLAlchemy Enum type,
  which avoids migration complexity when values are added.
- Every table has created_at (and resolved_at where relevant) as TIMESTAMPTZ
  so trace reconstruction is timezone-safe.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import text
from sqlalchemy.types import TIMESTAMP


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

class Prompt(Base):
    """
    Versioned prompt store. Each agent has exactly one active prompt at a time.
    The meta-agent reads the active prompt and writes a PromptRewrite proposal.

    Partial unique index WHERE is_active = TRUE enforces single-active-per-agent
    at the database level — defined in the SQL migration, not here, because
    SQLAlchemy's Index() cannot express partial indexes portably.
    """
    __tablename__ = "prompts"
    __table_args__ = (
        UniqueConstraint("agent_id", "version", name="uq_prompts_agent_version"),
    )

    id         = Column(UUID(as_uuid=True), primary_key=True,
                        server_default=text("gen_random_uuid()"))
    agent_id   = Column(Text, nullable=False, index=True)
    version    = Column(Integer, nullable=False, default=1)
    content    = Column(Text, nullable=False)
    is_active  = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        server_default=text("NOW()"))

    rewrites = relationship(
        "PromptRewrite",
        back_populates="original",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<Prompt agent={self.agent_id!r} v={self.version} active={self.is_active}>"


# ---------------------------------------------------------------------------
# Prompt Rewrites
# ---------------------------------------------------------------------------

class PromptRewrite(Base):
    """
    A proposed prompt rewrite from the meta-agent.
    Status transitions: pending → approved | rejected
    Approved rewrites trigger targeted re-evaluation; they are NEVER auto-applied.
    """
    __tablename__ = "prompt_rewrites"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_prompt_rewrites_status",
        ),
    )

    id               = Column(UUID(as_uuid=True), primary_key=True,
                              server_default=text("gen_random_uuid()"))
    original_id      = Column(UUID(as_uuid=True),
                              ForeignKey("prompts.id", ondelete="CASCADE"),
                              nullable=False)
    proposed_content = Column(Text, nullable=False)
    justification    = Column(Text)
    diff             = Column(Text)           # unified diff string
    status           = Column(Text, nullable=False, default="pending")
    approved_by      = Column(Text)           # human identifier
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False,
                              server_default=text("NOW()"))
    resolved_at      = Column(TIMESTAMP(timezone=True))

    original = relationship("Prompt", back_populates="rewrites")

    def __repr__(self) -> str:
        return f"<PromptRewrite id={self.id} status={self.status!r}>"


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

class Log(Base):
    """
    Structured event log. Every agent lifecycle event writes one row.
    All fields map directly to the logging schema in the architecture doc.

    event_type values (enforced by CHECK constraint in migration):
        agent_start, agent_complete, agent_error,
        tool_call, tool_result, retry,
        budget_violation, routing_decision, sse_emit,
        job_start, job_complete, job_error, sse_token

    input_hash / output_hash are SHA-256 hex digests of the serialised
    input/output — useful for deduplication and reproducibility checks.
    """
    __tablename__ = "logs"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ("
            "'agent_start','agent_complete','agent_error',"
            "'tool_call','tool_result','retry',"
            "'budget_violation','routing_decision','sse_emit',"
            "'job_start','job_complete','job_error','sse_token'"
            ")",
            name="ck_logs_event_type",
        ),
        Index("idx_logs_job_timestamp", "job_id", "timestamp"),
    )

    id                = Column(UUID(as_uuid=True), primary_key=True,
                               server_default=text("gen_random_uuid()"))
    timestamp         = Column(TIMESTAMP(timezone=True), nullable=False,
                               server_default=text("NOW()"))
    job_id            = Column(UUID(as_uuid=True), nullable=False, index=True)
    agent_id          = Column(Text, nullable=False)
    event_type        = Column(Text, nullable=False)
    latency_ms        = Column(Integer)        # NULL on agent_start
    token_count       = Column(Integer)
    input_hash        = Column(Text)
    output_hash       = Column(Text)
    policy_violations = Column(JSONB, nullable=False, default=list)
    payload           = Column(JSONB, nullable=False, default=dict)

    def __repr__(self) -> str:
        return (
            f"<Log job={self.job_id} agent={self.agent_id!r} "
            f"event={self.event_type!r}>"
        )


# ---------------------------------------------------------------------------
# Tool Calls
# ---------------------------------------------------------------------------

class ToolCall(Base):
    """
    Records every tool invocation including retries (attempt column).
    error_type maps to the three failure contracts:
        timeout | malformed_input | empty_result
    output is NULL when the call errored; error is NULL when it succeeded.
    """
    __tablename__ = "tool_calls"
    __table_args__ = (
        CheckConstraint(
            "error_type IN ('timeout', 'malformed_input', 'empty_result') "
            "OR error_type IS NULL",
            name="ck_tool_calls_error_type",
        ),
    )

    id          = Column(UUID(as_uuid=True), primary_key=True,
                         server_default=text("gen_random_uuid()"))
    job_id      = Column(UUID(as_uuid=True), nullable=False, index=True)
    agent_id    = Column(Text, nullable=False)
    tool_name   = Column(Text, nullable=False, index=True)
    input       = Column(JSONB, nullable=False, default=dict)
    output      = Column(JSONB)                # NULL on error
    error       = Column(Text)                 # NULL on success
    error_type  = Column(Text)                 # timeout | malformed_input | empty_result
    latency_ms  = Column(Integer)
    attempt     = Column(Integer, nullable=False, default=1)
    created_at  = Column(TIMESTAMP(timezone=True), nullable=False,
                         server_default=text("NOW()"))

    def __repr__(self) -> str:
        return (
            f"<ToolCall job={self.job_id} tool={self.tool_name!r} "
            f"attempt={self.attempt} error={self.error_type!r}>"
        )


# ---------------------------------------------------------------------------
# Eval Runs
# ---------------------------------------------------------------------------

class EvalRun(Base):
    """
    One row per test case execution within a batch.
    batch_id groups a complete 15-case evaluation run.

    scores JSONB shape:
    {
        "correctness":              {"score": 0.9, "justification": "..."},
        "citation_accuracy":        {"score": 1.0, "justification": "..."},
        "contradiction_resolution": {"score": 0.8, "justification": "..."},
        "tool_efficiency":          {"score": 0.7, "justification": "..."},
        "context_compliance":       {"score": 1.0, "justification": "..."},
        "critique_agreement":       {"score": 0.6, "justification": "..."}
    }

    prompt_versions JSONB shape:
    {"orchestrator": 1, "decomposition_agent": 2, "retrieval_agent": 1, ...}
    Snapshots which prompt version was active so diff comparisons are accurate.
    """
    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint(
            "test_category IN ('baseline', 'ambiguous', 'adversarial')",
            name="ck_eval_runs_category",
        ),
        Index("idx_eval_runs_created_at", "created_at"),
    )

    id              = Column(UUID(as_uuid=True), primary_key=True,
                             server_default=text("gen_random_uuid()"))
    job_id          = Column(UUID(as_uuid=True), nullable=False)
    batch_id        = Column(UUID(as_uuid=True), nullable=False, index=True)
    test_case_id    = Column(Text, nullable=False, index=True)
    test_category   = Column(Text, nullable=False)
    query           = Column(Text, nullable=False)
    answer          = Column(Text)
    ground_truth    = Column(Text)
    scores          = Column(JSONB, nullable=False, default=dict)
    prompt_versions = Column(JSONB, nullable=False, default=dict)
    created_at      = Column(TIMESTAMP(timezone=True), nullable=False,
                             server_default=text("NOW()"))

    def __repr__(self) -> str:
        return (
            f"<EvalRun batch={self.batch_id} case={self.test_case_id!r} "
            f"category={self.test_category!r}>"
        )