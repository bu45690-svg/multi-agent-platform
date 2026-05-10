"""
api/main.py

FastAPI application — 5 endpoints + SSE streaming + lifespan.

Endpoints
---------
POST /query                  — submit a query, stream the answer via SSE
GET  /trace/{job_id}         — replay all log events for a job
GET  /eval/latest            — latest eval run scores
POST /eval/run               — trigger a full eval harness run
POST /prompts/{id}/approve   — approve a pending prompt rewrite
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from logger.structured import get_logger

from worker.main import WorkerState, enqueue_job

# ---------------------------------------------------------------------------
# App state — single WorkerState shared via app.state
# ---------------------------------------------------------------------------

_worker_state = WorkerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _worker_state.startup()
    yield
    await _worker_state.shutdown()


app = FastAPI(
    title="Multi-Agent Orchestration Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB session dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    factory = _worker_state._session_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="Database not available")
    async with factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    job_id: str | None = None


class ApproveRequest(BaseModel):
    approved_by: str = "human"


# ---------------------------------------------------------------------------
# POST /query  — SSE streaming endpoint
# ---------------------------------------------------------------------------


@app.post("/query")
async def post_query(req: QueryRequest):
    """
    Submit a query. Returns a Server-Sent Events stream.
    """

    log = get_logger("query_orchestrator")

    # FIX 1: correct job creation (THIS WAS BROKEN)
    job_id, sse_queue = await enqueue_job(
        _worker_state,
        query=req.query,
        job_id=req.job_id,
    )

    # FIX 2: correct job start logging (DO NOT overwrite job system)
    asyncio.create_task(
        log.emit(
            job_id,
            "job_start",
            payload={"query": req.query},
        )
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: [JOB_ID] {job_id}\n\n"

        try:
            while True:
                token = await asyncio.wait_for(sse_queue.get(), timeout=120.0)

                if token is None:
                    yield "data: [DONE]\n\n"

                    asyncio.create_task(
                        log.emit(job_id, "job_complete")
                    )
                    break

                # FIX 3: correct logging (non-blocking)
                asyncio.create_task(
                    log.emit(
                        job_id,
                        "sse_token",
                        payload={"token": token}
                    )
                )

                yield f"data: {token}\n\n"

        except asyncio.TimeoutError:
            yield "data: [ERROR] Stream timed out\n\n"

            asyncio.create_task(
                log.emit(job_id, "job_error", payload={"reason": "timeout"})
            )

        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

            asyncio.create_task(
                log.emit(job_id, "job_error", payload={"error": str(exc)})
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Job-ID": job_id,
        },
    )

# ---------------------------------------------------------------------------
# GET /trace/{job_id}
# ---------------------------------------------------------------------------


@app.get("/trace/{job_id}")
async def get_trace(
    job_id: str = Path(..., description="Job UUID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return all log events for a job, ordered by timestamp.
    Reconstructs the full pipeline replay.
    """
    rows = await db.execute(
        text(
            """
            SELECT
                timestamp, agent_id, event_type, latency_ms,
                token_count, policy_violations, payload
            FROM logs
            WHERE job_id = :job_id
            ORDER BY timestamp ASC
            """
        ),
        {"job_id": job_id},
    )
    events = [dict(row._mapping) for row in rows.fetchall()]

    if not events:
        raise HTTPException(status_code=404, detail=f"No trace found for job {job_id}")

    tool_rows = await db.execute(
        text(
            """
            SELECT
                tool_name, input, output, error,
                latency_ms, attempt, created_at
            FROM tool_calls
            WHERE job_id = :job_id
            ORDER BY created_at ASC
            """
        ),
        {"job_id": job_id},
    )
    tool_calls = [dict(row._mapping) for row in tool_rows.fetchall()]

    return {
        "job_id":     job_id,
        "event_count": len(events),
        "events":     events,
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# GET /eval/latest
# ---------------------------------------------------------------------------


@app.get("/eval/latest")
async def get_eval_latest(db: AsyncSession = Depends(get_db)):
    """
    Return the most recent eval run results, aggregated by test category.
    """
    rows = await db.execute(
        text(
            """
            SELECT
                test_category,
                COUNT(*)                          AS test_count,
                AVG((scores->>'correctness')::float)       AS avg_correctness,
                AVG((scores->>'citation_accuracy')::float) AS avg_citation,
                AVG((scores->>'tool_efficiency')::float)   AS avg_tool_efficiency,
                MAX(created_at)                   AS latest_run_at
            FROM eval_runs
            WHERE created_at = (SELECT MAX(created_at) FROM eval_runs)
            GROUP BY test_category
            ORDER BY test_category
            """
        )
    )
    categories = [dict(row._mapping) for row in rows.fetchall()]

    if not categories:
        raise HTTPException(status_code=404, detail="No eval runs found")

    # Overall aggregates
    overall = await db.execute(
        text(
            """
            SELECT
                COUNT(*)                                      AS total_tests,
                AVG((scores->>'correctness')::float)          AS avg_correctness,
                AVG((scores->>'citation_accuracy')::float)    AS avg_citation,
                AVG((scores->>'contradiction_resolution')::float) AS avg_contradiction,
                AVG((scores->>'tool_efficiency')::float)      AS avg_tool_efficiency,
                AVG((scores->>'context_compliance')::float)   AS avg_context_compliance,
                AVG((scores->>'critique_agreement')::float)   AS avg_critique_agreement
            FROM eval_runs
            WHERE created_at >= (
                SELECT MAX(created_at) - INTERVAL '1 minute' FROM eval_runs
            )
            """
        )
    )
    overall_row = dict(overall.fetchone()._mapping)

    return {
        "by_category": categories,
        "overall":     overall_row,
    }


# ---------------------------------------------------------------------------
# POST /eval/run
# ---------------------------------------------------------------------------


@app.post("/eval/run")
async def post_eval_run():
    """
    Trigger a full eval harness run asynchronously.
    Returns immediately with a run_id; results appear in /eval/latest.
    """
    import uuid as _uuid
    run_id = str(_uuid.uuid4())

    async def _run_eval():
        try:
            from eval.harness import EvalHarness
            harness = EvalHarness(worker_state=_worker_state)
            await harness.run(run_id=run_id)
        except Exception as exc:
            import logging
            logging.getLogger("eval").exception("Eval run %s failed: %s", run_id, exc)

    asyncio.create_task(_run_eval(), name=f"eval_{run_id}")

    return {"run_id": run_id, "status": "started"}


# ---------------------------------------------------------------------------
# POST /prompts/{id}/approve
# ---------------------------------------------------------------------------


@app.post("/prompts/{rewrite_id}/approve")
async def approve_prompt(
    rewrite_id: str = Path(..., description="prompt_rewrites.id UUID"),
    req: ApproveRequest = ApproveRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Approve a pending prompt rewrite.

    1. Marks the prompt_rewrites row as 'approved'.
    2. Inserts a new prompts row (version + 1) with is_active = TRUE.
    3. Deactivates the previous version.
    """
    # Fetch the rewrite
    row = await db.execute(
        text(
            "SELECT * FROM prompt_rewrites WHERE id = :id AND status = 'pending'"
        ),
        {"id": rewrite_id},
    )
    rewrite = row.fetchone()
    if not rewrite:
        raise HTTPException(
            status_code=404,
            detail=f"Pending rewrite {rewrite_id} not found",
        )
    rewrite = dict(rewrite._mapping)

    # Determine agent_id from the original prompt
    orig_row = await db.execute(
        text("SELECT agent_id, version FROM prompts WHERE id = :id"),
        {"id": rewrite["original_id"]},
    )
    original = orig_row.fetchone()
    if not original:
        raise HTTPException(status_code=404, detail="Original prompt not found")
    original = dict(original._mapping)

    agent_id = original["agent_id"]
    new_version = original["version"] + 1

    # Deactivate old prompts for this agent
    await db.execute(
        text("UPDATE prompts SET is_active = FALSE WHERE agent_id = :aid"),
        {"aid": agent_id},
    )

    # Insert new active prompt
    await db.execute(
        text(
            """
            INSERT INTO prompts (agent_id, version, content, is_active)
            VALUES (:aid, :ver, :content, TRUE)
            """
        ),
        {
            "aid":     agent_id,
            "ver":     new_version,
            "content": rewrite["proposed_content"],
        },
    )

    # Mark rewrite as approved
    await db.execute(
        text(
            """
            UPDATE prompt_rewrites
            SET status = 'approved', approved_by = :by
            WHERE id = :id
            """
        ),
        {"by": req.approved_by, "id": rewrite_id},
    )

    await db.commit()

    return {
        "rewrite_id":  rewrite_id,
        "agent_id":    agent_id,
        "new_version": new_version,
        "approved_by": req.approved_by,
        "status":      "approved",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "queue_size": _worker_state.job_queue.qsize(),
    }