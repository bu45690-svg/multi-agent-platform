"""
db/schemas.py
Pydantic schemas for database objects.

Separates API-layer validation from ORM models.
Every schema has a Config with from_attributes=True so it can be built
directly from a SQLAlchemy ORM row via model_validate(row).

Naming convention:
    <Model>Base   — shared fields
    <Model>Create — fields required to create a row (no server-generated fields)
    <Model>Read   — full read schema including server-generated fields
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

class _OrmBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

class PromptCreate(BaseModel):
    agent_id : str
    version  : int = 1
    content  : str
    is_active: bool = True


class PromptRead(_OrmBase):
    id        : uuid.UUID
    agent_id  : str
    version   : int
    content   : str
    is_active : bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Prompt Rewrite
# ---------------------------------------------------------------------------

class PromptRewriteCreate(BaseModel):
    original_id     : uuid.UUID
    proposed_content: str
    justification   : str | None = None
    diff            : str | None = None


class PromptRewriteRead(_OrmBase):
    id              : uuid.UUID
    original_id     : uuid.UUID
    proposed_content: str
    justification   : str | None
    diff            : str | None
    status          : str
    approved_by     : str | None
    created_at      : datetime
    resolved_at     : datetime | None


class PromptApprovalRequest(BaseModel):
    """Body for POST /prompts/{rewrite_id}/approve or /reject."""
    approved_by: str = Field(..., description="Human identifier (name or email)")
    action     : str = Field(..., pattern="^(approved|rejected)$")


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

class LogCreate(BaseModel):
    job_id           : uuid.UUID
    agent_id         : str
    event_type       : str
    latency_ms       : int | None = None
    token_count      : int | None = None
    input_hash       : str | None = None
    output_hash      : str | None = None
    policy_violations: list[str] = []
    payload          : dict[str, Any] = {}


class LogRead(_OrmBase):
    id               : uuid.UUID
    timestamp        : datetime
    job_id           : uuid.UUID
    agent_id         : str
    event_type       : str
    latency_ms       : int | None
    token_count      : int | None
    input_hash       : str | None
    output_hash      : str | None
    policy_violations: list[str]
    payload          : dict[str, Any]


# ---------------------------------------------------------------------------
# Tool Call
# ---------------------------------------------------------------------------

class ToolCallCreate(BaseModel):
    job_id    : uuid.UUID
    agent_id  : str
    tool_name : str
    input     : dict[str, Any]
    output    : dict[str, Any] | None = None
    error     : str | None = None
    error_type: str | None = None          # timeout | malformed_input | empty_result
    latency_ms: int | None = None
    attempt   : int = 1


class ToolCallRead(_OrmBase):
    id        : uuid.UUID
    job_id    : uuid.UUID
    agent_id  : str
    tool_name : str
    input     : dict[str, Any]
    output    : dict[str, Any] | None
    error     : str | None
    error_type: str | None
    latency_ms: int | None
    attempt   : int
    created_at: datetime


# ---------------------------------------------------------------------------
# Eval Run
# ---------------------------------------------------------------------------

class ScoreDetail(BaseModel):
    """A single scored dimension."""
    score        : float = Field(..., ge=0.0, le=1.0)
    justification: str


class EvalScores(BaseModel):
    """
    All six scoring dimensions. Fields are optional so partial results
    can be stored during a run that errored mid-way.
    """
    correctness             : ScoreDetail | None = None
    citation_accuracy       : ScoreDetail | None = None
    contradiction_resolution: ScoreDetail | None = None
    tool_efficiency         : ScoreDetail | None = None
    context_compliance      : ScoreDetail | None = None
    critique_agreement      : ScoreDetail | None = None

    def mean(self) -> float:
        """Return mean of all present scores."""
        values = [
            v.score for v in [
                self.correctness, self.citation_accuracy,
                self.contradiction_resolution, self.tool_efficiency,
                self.context_compliance, self.critique_agreement,
            ]
            if v is not None
        ]
        return sum(values) / len(values) if values else 0.0


class EvalRunCreate(BaseModel):
    job_id         : uuid.UUID
    batch_id       : uuid.UUID
    test_case_id   : str
    test_category  : str
    query          : str
    answer         : str | None = None
    ground_truth   : str | None = None
    scores         : dict[str, Any] = {}
    prompt_versions: dict[str, int] = {}


class EvalRunRead(_OrmBase):
    id             : uuid.UUID
    job_id         : uuid.UUID
    batch_id       : uuid.UUID
    test_case_id   : str
    test_category  : str
    query          : str
    answer         : str | None
    ground_truth   : str | None
    scores         : dict[str, Any]
    prompt_versions: dict[str, int]
    created_at     : datetime


# ---------------------------------------------------------------------------
# Execution Trace (assembled from logs — not a DB table)
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """Single event in a reconstructed execution trace."""
    timestamp        : datetime
    agent_id         : str
    event_type       : str
    latency_ms       : int | None
    token_count      : int | None
    policy_violations: list[str]
    payload          : dict[str, Any]


class ExecutionTrace(BaseModel):
    """Full trace for a job_id, returned by GET /trace/{job_id}."""
    job_id          : uuid.UUID
    events          : list[TraceEvent]
    tool_calls      : list[ToolCallRead]
    total_tokens    : int
    total_latency_ms: int


# ---------------------------------------------------------------------------
# Eval Summary (assembled across batch — not a DB table)
# ---------------------------------------------------------------------------

class EvalBatchSummary(BaseModel):
    """Returned by GET /eval/latest."""
    batch_id    : uuid.UUID
    created_at  : datetime
    total_cases : int
    by_category : dict[str, int]      # {baseline: 5, ambiguous: 5, adversarial: 5}
    mean_scores : dict[str, float]    # {correctness: 0.84, citation_accuracy: 0.91, ...}
    worst_agent : str | None          # agent_id with lowest aggregate score
    runs        : list[EvalRunRead]