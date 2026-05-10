"""
eval/harness.py

Evaluation harness — runs all 15 test cases through the live pipeline,
scores them with all 6 scorers, persists results to eval_runs, and
prints a summary table.

Usage
-----
Standalone:
    python -m eval.harness

Via API:
    POST /eval/run  → triggers EvalHarness.run() as a background task
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

from eval.scoring import score_all, scores_to_dict
from eval.test_cases import TEST_CASES, TestCase

if TYPE_CHECKING:
    from worker.main import WorkerState

log = logging.getLogger("eval.harness")

# Run test cases concurrently in batches to avoid overwhelming the worker
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "3"))


class EvalHarness:
    """
    Parameters
    ----------
    worker_state : WorkerState — the live worker (provides enqueue_job)
    db_session_factory : optional override; falls back to worker_state's factory
    """

    def __init__(
        self,
        worker_state: "WorkerState | None" = None,
        db_session_factory=None,
    ) -> None:
        self._worker_state = worker_state
        self._db = db_session_factory or (
            worker_state._session_factory if worker_state else None
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, run_id: str | None = None) -> list[dict]:
        """
        Run all 15 test cases, score them, persist to DB, and return results.
        """
        run_id = run_id or str(uuid.uuid4())
        log.info("Eval run %s starting — %d test cases", run_id, len(TEST_CASES))
        t0 = time.perf_counter()

        results: list[dict] = []
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run_one(tc: TestCase) -> dict:
            async with sem:
                return await self._run_case(tc, run_id)

        tasks = [run_one(tc) for tc in TEST_CASES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions — log them but keep going
        clean: list[dict] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                log.error("Test case %s failed: %s", TEST_CASES[i].id, r)
            else:
                clean.append(r)

        elapsed = time.perf_counter() - t0
        self._print_summary(clean, elapsed)
        return clean

    # ------------------------------------------------------------------
    # Single test case
    # ------------------------------------------------------------------

    async def _run_case(self, tc: TestCase, run_id: str) -> dict:
        log.info("Running %s [%s]: %s…", tc.id, tc.category, tc.query[:60])
        t0 = time.perf_counter()

        # ── Submit to worker ───────────────────────────────────────────
        if self._worker_state is not None:
            from worker.main import enqueue_job
            job_id, sse_queue = await enqueue_job(
                self._worker_state, query=tc.query
            )
            # Drain SSE (we don't need the streaming for eval)
            while True:
                token = await asyncio.wait_for(sse_queue.get(), timeout=120)
                if token is None:
                    break

            # Retrieve context from the orchestrator via the result future
            # (WorkerState._drain_queue resolves it)
            from worker.context import SharedContext
            context = await self._fetch_context_from_db(job_id)
        else:
            # Standalone mode: run orchestrator directly
            from worker.orchestrator import Orchestrator
            from worker.rag.indexer import build_or_load_index
            from worker.rag.retriever import RAGRetriever

            try:
                index, store = await build_or_load_index()
                retriever = RAGRetriever(index, store)
            except Exception:
                retriever = None

            orch = Orchestrator(retriever=retriever, db_session_factory=self._db)
            context = await orch.run(query=tc.query)
            job_id = context.job_id

        # ── Score ──────────────────────────────────────────────────────
        score_results = await score_all(tc, context)
        scores = scores_to_dict(score_results)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # ── Persist to eval_runs ───────────────────────────────────────
        await self._persist(
            run_id=run_id,
            job_id=job_id,
            tc=tc,
            scores=scores,
            answer=context.final_answer,
            context=context,
        )

        result = {
            "test_id":    tc.id,
            "category":   tc.category,
            "job_id":     job_id,
            "scores":     scores,
            "latency_ms": latency_ms,
            "answer_len": len(context.final_answer or ""),
        }
        log.info(
            "  %s → avg=%.2f  lat=%dms",
            tc.id,
            sum(scores.values()) / len(scores),
            latency_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Context retrieval (from DB snapshot when running via worker)
    # ------------------------------------------------------------------

    async def _fetch_context_from_db(self, job_id: str):
        """
        Fetch the SharedContext snapshot persisted by the Orchestrator.
        Falls back to a minimal stub if DB is unavailable.
        """
        from worker.context import SharedContext
        if self._db is None:
            return SharedContext(job_id=job_id, query="")

        try:
            from sqlalchemy import text
            async with self._db() as session:
                row = await session.execute(
                    text(
                        """
                        SELECT payload FROM logs
                        WHERE job_id = :jid AND event_type = 'job_complete'
                        ORDER BY timestamp DESC LIMIT 1
                        """
                    ),
                    {"jid": job_id},
                )
                result = row.fetchone()
                if result:
                    data = result[0]
                    if isinstance(data, str):
                        data = json.loads(data)
                    return SharedContext.from_snapshot(data)
        except Exception as exc:
            log.warning("Could not fetch context for job %s: %s", job_id, exc)

        return SharedContext(job_id=job_id, query="")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(
        self,
        run_id: str,
        job_id: str,
        tc: TestCase,
        scores: dict[str, float],
        answer: str | None,
        context,
    ) -> None:
        if self._db is None:
            return
        try:
            from sqlalchemy import text
            async with self._db() as session:
                # Snapshot the prompt versions used (best-effort)
                prompt_versions = await self._collect_prompt_versions(session)
                await session.execute(
                    text(
                        """
                        INSERT INTO eval_runs
                            (job_id, test_case_id, test_category,
                             scores, answer, prompt_versions)
                        VALUES
                            (:job_id, :test_case_id, :test_category,
                             :scores::jsonb, :answer, :prompt_versions::jsonb)
                        """
                    ),
                    {
                        "job_id":          job_id,
                        "test_case_id":    tc.id,
                        "test_category":   tc.category,
                        "scores":          json.dumps(scores),
                        "answer":          answer,
                        "prompt_versions": json.dumps(prompt_versions),
                    },
                )
                await session.commit()
        except Exception as exc:
            log.warning("eval_runs persist failed: %s", exc)

    async def _collect_prompt_versions(self, session) -> dict:
        try:
            from sqlalchemy import text
            rows = await session.execute(
                text(
                    "SELECT agent_id, version FROM prompts WHERE is_active = TRUE"
                )
            )
            return {r.agent_id: r.version for r in rows.fetchall()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def _print_summary(self, results: list[dict], elapsed: float) -> None:
        if not results:
            log.info("No results to summarise.")
            return

        scorer_names = [
            "correctness", "citation_accuracy", "contradiction_resolution",
            "tool_efficiency", "context_compliance", "critique_agreement",
        ]

        # Group by category
        by_cat: dict[str, list[dict]] = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r)

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            f"║  EVAL RESULTS   ({len(results)}/15 cases)   {elapsed:.1f}s total             ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        for cat, cat_results in sorted(by_cat.items()):
            lines.append(f"║  Category: {cat:<50}║")
            for r in cat_results:
                s = r["scores"]
                avg = sum(s.values()) / len(s)
                lines.append(
                    f"║    {r['test_id']:<18} avg={avg:.2f}  "
                    f"corr={s.get('correctness', 0):.2f}  "
                    f"cit={s.get('citation_accuracy', 0):.2f}  "
                    f"lat={r['latency_ms']}ms  ║"
                )
            cat_avg = {
                name: sum(r["scores"].get(name, 0) for r in cat_results) / len(cat_results)
                for name in scorer_names
            }
            lines.append(
                f"║  → {cat} avg: "
                + "  ".join(f"{n[:4]}={v:.2f}" for n, v in cat_avg.items())
                + "  ║"
            )
            lines.append("║" + "─" * 62 + "║")

        # Overall
        overall = {
            name: sum(r["scores"].get(name, 0) for r in results) / len(results)
            for name in scorer_names
        }
        lines.append(f"║  OVERALL ({len(results)} cases):{' ' * 42}║")
        for name, val in overall.items():
            lines.append(f"║    {name:<35} {val:.3f}                  ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        for line in lines:
            log.info(line)
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    async def main():
        harness = EvalHarness()
        await harness.run()

    asyncio.run(main())