"""
worker/main.py

Async worker entry point.

Lifecycle
---------
1. On startup: build/load FAISS index, wire up DB session factory, configure
   the structured logger, instantiate the Orchestrator.
2. Spin up a background task that drains the in-process asyncio.Queue.
3. On each job: create a per-job SSE queue, run Orchestrator.run(), push
   the final answer back through the SSE queue.
4. On shutdown: drain the queue gracefully, close DB connections.

The worker runs *inside the FastAPI process* as a background task started
by the FastAPI lifespan.  This keeps the architecture Redis-free — the job
queue is a plain asyncio.Queue shared between the request handlers and the
worker coroutine.

Public interface (used by api/main.py)
---------------------------------------
    from worker.main import WorkerState, enqueue_job, startup, shutdown

    # In FastAPI lifespan:
    state = WorkerState()
    await state.startup()
    yield
    await state.shutdown()

    # In route handler:
    job_id, sse_q = await enqueue_job(state, query="What is RAG?")
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import logger.structured as structured_log
from worker.orchestrator import Orchestrator
from worker.rag.indexer import build_or_load_index

log = logging.getLogger("worker.main")

# ---------------------------------------------------------------------------
# Job envelope
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: str
    query: str
    sse_queue: asyncio.Queue          # tokens pushed here by SynthesisAgent
    result_future: asyncio.Future     # resolved with SharedContext when done


# ---------------------------------------------------------------------------
# WorkerState — one instance shared across the FastAPI app
# ---------------------------------------------------------------------------


@dataclass
class WorkerState:
    job_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    orchestrator: Orchestrator | None = None
    _worker_task: asyncio.Task | None = field(default=None, repr=False)
    _session_factory: Any | None = field(default=None, repr=False)
    _engine: Any | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Called from FastAPI lifespan on startup.
        Initialises DB, logger, FAISS index, and the Orchestrator.
        """
        log.info("Worker starting up…")

        # 1. Database engine + session factory
        db_url = os.environ.get("DATABASE_URL", "")
        if db_url:
            # asyncpg driver — replace postgres:// with postgresql+asyncpg://
            async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            self._engine = create_async_engine(async_url, pool_size=5, echo=False)
            self._session_factory = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
            structured_log.configure(db_session_factory=self._session_factory)
            log.info("Database connected.")
        else:
            log.warning("DATABASE_URL not set — running without DB persistence.")

        # 2. Build / load FAISS index
        try:
            index, store = await build_or_load_index()
            from worker.rag.retriever import RAGRetriever
            retriever = RAGRetriever(index, store)
            log.info("FAISS index ready (%d chunks).", len(store))
        except Exception as exc:
            log.warning("FAISS index unavailable (%s) — retrieval will web_search only.", exc)
            retriever = None

        # 3. Orchestrator
        self.orchestrator = Orchestrator(
            retriever=retriever,
            db_session_factory=self._session_factory,
        )

        # 4. Start background worker loop
        self._worker_task = asyncio.create_task(
            self._drain_queue(), name="worker_drain_queue"
        )
        log.info("Worker started.")

    async def shutdown(self) -> None:
        """Called from FastAPI lifespan on shutdown."""
        log.info("Worker shutting down…")
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._engine:
            await self._engine.dispose()
        log.info("Worker stopped.")

    # ------------------------------------------------------------------
    # Queue drain loop
    # ------------------------------------------------------------------

    async def _drain_queue(self) -> None:
        """
        Continuously drain the job queue and dispatch each job to the
        Orchestrator.  Runs as a long-lived background task.
        """
        while True:
            job: Job = await self.job_queue.get()
            try:
                context = await self.orchestrator.run(
                    query=job.query,
                    job_id=job.job_id,
                    sse_queue=job.sse_queue,
                )
                job.result_future.set_result(context)
            except Exception as exc:
                log.exception("Job %s failed: %s", job.job_id, exc)
                if not job.result_future.done():
                    job.result_future.set_exception(exc)
                # Ensure SSE stream is closed so clients don't hang
                try:
                    await job.sse_queue.put(None)
                except Exception:
                    pass
            finally:
                self.job_queue.task_done()


# ---------------------------------------------------------------------------
# Public helper used by route handlers
# ---------------------------------------------------------------------------


async def enqueue_job(
    state: WorkerState,
    query: str,
    job_id: str | None = None,
) -> tuple[str, asyncio.Queue]:
    """
    Submit a query to the worker queue.

    Returns
    -------
    (job_id, sse_queue)  — route handler streams tokens from sse_queue
    """
    job_id = job_id or str(uuid.uuid4())
    sse_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    result_future: asyncio.Future = loop.create_future()

    job = Job(
        job_id=job_id,
        query=query,
        sse_queue=sse_queue,
        result_future=result_future,
    )
    await state.job_queue.put(job)
    return job_id, sse_queue