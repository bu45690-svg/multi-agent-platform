-- =============================================================================
-- Migration 001 — Initial Schema
-- Multi-Agent Orchestration Platform
-- =============================================================================

-- Enable UUID generation (available in Postgres 13+ without extension,
-- but we keep gen_random_uuid() which requires pgcrypto on older versions)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- TABLE: prompts
-- Stores the active prompt for each agent, versioned.
-- One row per (agent_id, version). The orchestrator always loads is_active=TRUE.
-- =============================================================================
CREATE TABLE IF NOT EXISTS prompts (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT        NOT NULL,
    version     INT         NOT NULL DEFAULT 1,
    content     TEXT        NOT NULL,
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_prompts_agent_version UNIQUE (agent_id, version)
);

-- Only one active prompt per agent at a time (partial unique index)
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompts_agent_active
    ON prompts (agent_id)
    WHERE is_active = TRUE;

-- =============================================================================
-- TABLE: prompt_rewrites
-- Stores proposed prompt improvements from the meta-agent.
-- Status lifecycle: pending → approved | rejected
-- Approved rewrites trigger targeted re-evaluation (never auto-applied).
-- =============================================================================
CREATE TABLE IF NOT EXISTS prompt_rewrites (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    original_id      UUID        NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    proposed_content TEXT        NOT NULL,
    justification    TEXT,
    diff             TEXT,                          -- unified diff string
    status           TEXT        NOT NULL DEFAULT 'pending'
                                 CHECK (status IN ('pending', 'approved', 'rejected')),
    approved_by      TEXT,                          -- human identifier
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status
    ON prompt_rewrites (status);

CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_original
    ON prompt_rewrites (original_id);

-- =============================================================================
-- TABLE: logs
-- Structured event log. One row per agent event (start, complete, error, etc.)
-- This is the source of truth for execution trace reconstruction.
-- =============================================================================
CREATE TABLE IF NOT EXISTS logs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id            UUID        NOT NULL,
    agent_id          TEXT        NOT NULL,
    event_type        TEXT        NOT NULL
                      CHECK (event_type IN (
                          'agent_start',
                          'agent_complete',
                          'agent_error',
                          'tool_call',
                          'tool_result',
                          'retry',
                          'budget_violation',
                          'routing_decision',
                          'sse_emit',
                          'job_start',
                          'job_complete',
                          'job_error',
                          'sse_token'
                      )),
    latency_ms        INT,                          -- NULL for agent_start events
    token_count       INT,
    input_hash        TEXT,                         -- SHA-256 of serialized input
    output_hash       TEXT,                         -- SHA-256 of serialized output
    policy_violations JSONB       NOT NULL DEFAULT '[]'::jsonb,
    payload           JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_logs_job_id
    ON logs (job_id);

CREATE INDEX IF NOT EXISTS idx_logs_job_timestamp
    ON logs (job_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_logs_event_type
    ON logs (event_type);

-- =============================================================================
-- TABLE: tool_calls
-- One row per tool invocation. Tracks retries via `attempt` column.
-- Separate from logs so tool performance can be queried independently.
-- =============================================================================
CREATE TABLE IF NOT EXISTS tool_calls (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      UUID        NOT NULL,
    agent_id    TEXT        NOT NULL,
    tool_name   TEXT        NOT NULL,
    input       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    output      JSONB,                              -- NULL if errored
    error       TEXT,                               -- NULL if successful
    error_type  TEXT                                -- timeout | malformed_input | empty_result
                CHECK (
                    error_type IN ('timeout', 'malformed_input', 'empty_result')
                    OR error_type IS NULL
                ),
    latency_ms  INT,
    attempt     INT         NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_job_id
    ON tool_calls (job_id);

CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name
    ON tool_calls (tool_name);

-- =============================================================================
-- TABLE: eval_runs
-- One row per (test case × eval batch). Stores all scores as JSONB so scoring
-- dimensions can evolve without schema changes.
-- prompt_versions snapshots which prompt version was active during the run,
-- enabling accurate diff comparisons.
-- =============================================================================
CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID        NOT NULL,
    batch_id        UUID        NOT NULL,           -- groups a full 15-case run
    test_case_id    TEXT        NOT NULL,
    test_category   TEXT        NOT NULL
                    CHECK (test_category IN ('baseline', 'ambiguous', 'adversarial')),
    query           TEXT        NOT NULL,
    answer          TEXT,
    ground_truth    TEXT,
    scores          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    -- Shape: {
    --   "correctness":              {"score": 0.9, "justification": "..."},
    --   "citation_accuracy":        {"score": 1.0, "justification": "..."},
    --   "contradiction_resolution": {...},
    --   "tool_efficiency":          {...},
    --   "context_compliance":       {...},
    --   "critique_agreement":       {...}
    -- }
    prompt_versions JSONB       NOT NULL DEFAULT '{}'::jsonb,
    -- Shape: {"decomposition": 2, "retrieval": 1, ...}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_batch_id
    ON eval_runs (batch_id);

CREATE INDEX IF NOT EXISTS idx_eval_runs_test_case_id
    ON eval_runs (test_case_id);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at
    ON eval_runs (created_at DESC);

-- =============================================================================
-- SEED: Default prompts for each agent (version 1, active)
-- These are intentionally minimal — the meta-agent will propose improvements.
-- =============================================================================
INSERT INTO prompts (agent_id, version, content, is_active) VALUES
(
    'orchestrator', 1,
    'You are an orchestration agent. Given a user query, decide which agents '
    'to run and in what order. Respond with a JSON object: '
    '{"agents": [...], "budgets": {...}, "reasoning": "..."}. '
    'Available agents: decomposition_agent, retrieval_agent, critique_agent, synthesis_agent. '
    'Always include all four unless the query is trivially simple. '
    'Total token budget must not exceed 12000.',
    TRUE
),
(
    'decomposition_agent', 1,
    'You are a decomposition agent. Break the user query into a structured '
    'task DAG. Respond ONLY with a JSON array of task nodes. '
    'Each node: {"task_id": "t1", "task_type": "retrieve|compute|synthesize", '
    '"description": "...", "dependencies": []}. '
    'The final task must be type synthesize. Maximum 6 tasks.',
    TRUE
),
(
    'retrieval_agent', 1,
    'You are a retrieval agent. Given retrieved document chunks and a query, '
    'extract factual claims. For each claim, identify the source chunk. '
    'Respond with a JSON array: [{"text": "...", "confidence": 0.9}]. '
    'Maximum 5 claims per passage. Return [] if no relevant facts found.',
    TRUE
),
(
    'critique_agent', 1,
    'You are a critique agent. Review each claim individually. '
    'Score confidence, flag unsupported spans, and detect contradictions. '
    'Do NOT assess the answer globally — evaluate each claim in isolation. '
    'Respond with JSON: {"overall_score": 0.8, "criterion_scores": {...}, '
    '"issues": [...], "suggestions": [...], "verdict": "pass|warn|fail", '
    '"reasoning": "..."}.',
    TRUE
),
(
    'synthesis_agent', 1,
    'You are a synthesis agent. Using the task DAG, retrieved chunks, and '
    'critique results, write a final answer in clear prose (2-5 paragraphs). '
    'Mark DISPUTED claims explicitly. Cite sources inline as [chunk_id]. '
    'After your answer output: ---PROVENANCE--- '
    '<sentence excerpt> | <source_agent> | <chunk_id or none> ---END---',
    TRUE
),
(
    'meta_agent', 1,
    'You are a prompt engineer improving other agent prompts. '
    'Given a current prompt, failure diagnosis, and bad output examples, '
    'write an improved system prompt that fixes the diagnosed issues. '
    'Keep the same intent and output format. Output ONLY the new prompt text.',
    TRUE
)
ON CONFLICT (agent_id, version) DO NOTHING;