"""
worker/agents/retrieval.py
Fixed version — same architecture, only bug fixes applied.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from worker.agents.base import BaseAgent
from worker.context import Claim, RetrievedChunk, SharedContext
from worker.tools.base import ToolInput, ToolExecutionError, ToolTimeoutError
from worker.tools.web_search import WebSearchTool
from worker.tools.python_sandbox import PythonSandboxTool

if TYPE_CHECKING:
    from worker.rag.retriever import RAGRetriever

MIN_FAISS_SCORE = 0.30
MAX_CHUNKS_PER_TASK = 4
MAX_WEB_RESULTS = 3


_CLAIM_EXTRACTION_SYSTEM = """\
You are an information extractor. Extract key factual claims.

Return ONLY JSON array:
[
  {"text": "...", "confidence": 0.0-1.0}
]

Rules:
- max 5 claims
- no prose
- return [] if nothing found
"""


class RetrievalAgent(BaseAgent):
    agent_id = "retrieval_agent"
    max_tokens = 512

    def __init__(self, retriever=None, db_session_factory=None) -> None:
        super().__init__(db_session_factory)
        self._retriever = retriever
        self._web_tool = WebSearchTool(logger=self._log, db_session_factory=db_session_factory)
        self._sandbox_tool = PythonSandboxTool(logger=self._log, db_session_factory=db_session_factory)

    def _default_prompt(self) -> str:
        return _CLAIM_EXTRACTION_SYSTEM

    async def _validate(self, context: SharedContext) -> None:
        # FIX: ensure task_dag always iterable objects, not dict fallback breaking pipeline
        if not context.task_dag:
            context.task_dag = []

    # -----------------------------
    # MAIN EXECUTION
    # -----------------------------
    async def _execute(self, context: SharedContext) -> None:
        system = await self._load_prompt()

        retrieve_tasks = [
            t for t in context.task_dag
            if hasattr(t, "task_type")
            and getattr(t, "task_type", None) in ("retrieve", "compute")
            and getattr(t, "status", "pending") == "pending"
        ]

        for task in retrieve_tasks:
            try:
                task.mark_running()
                await self._handle_task(task, context, system)
                task.mark_done(output={"chunks_added": len(context.retrieved_chunks)})

            except Exception as exc:
                task.mark_failed(str(exc))
                await self._log.emit(
                    context.job_id,
                    "agent_error",
                    payload={"task_id": getattr(task, "task_id", None), "error": str(exc)},
                )

    # -----------------------------
    # TASK HANDLER
    # -----------------------------
    async def _handle_task(self, task, context: SharedContext, system: str) -> None:

        if task.task_type == "compute":
            await self._handle_compute(task, context)
            return

        chunks: list[RetrievedChunk] = []

        # -----------------------------
        # FIX 1: SAFE FAISS CALL
        # -----------------------------
        if self._retriever is not None:
            try:
                chunks = await self._retriever.retrieve_for_task(
                    task_description=task.description,
                    context_query=context.query,
                    top_k=MAX_CHUNKS_PER_TASK,
                    min_score=MIN_FAISS_SCORE,
                ) or []
            except Exception as exc:
                await self._log.emit(
                    context.job_id,
                    "retriever_error",
                    payload={"error": str(exc), "task_id": task.task_id},
                )
                chunks = []

        # -----------------------------
        # FIX 2: SAFE FALLBACK (no score dependency crash)
        # -----------------------------
        if not chunks:
            web_chunks = await self._web_search_fallback(task, context)
            chunks.extend(web_chunks)

        if not chunks:
            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={
                    "task_id": task.task_id,
                    "warning": "No chunks retrieved",
                },
            )
            return

        # -----------------------------
        # ADD TO CONTEXT
        # -----------------------------
        for chunk in chunks:
            context.add_chunk(chunk)

        # -----------------------------
        # CLAIM EXTRACTION
        # -----------------------------
        for chunk in chunks:
            claims = await self._extract_claims(chunk, context, system)
            for claim in claims:
                context.add_claim(claim)

    # -----------------------------
    # COMPUTE TASKS
    # -----------------------------
    async def _handle_compute(self, task, context: SharedContext) -> None:

        import re

        code_prompt = (
            f"Write Python code to compute:\n{task.description}\n"
            f"Print only final result."
        )

        code = await self._call_llm(
            context,
            [{"role": "user", "content": code_prompt}],
        )

        fence = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
        if fence:
            code = fence.group(1)

        inp = ToolInput(
            data={"code": code.strip(), "timeout_s": 10},
            job_id=context.job_id,
            agent_id=self.agent_id,
        )

        try:
            result = await self._sandbox_tool.execute(inp, context)
        except (ToolExecutionError, ToolTimeoutError):
            return

        if result.success:
            stdout = (result.output.get("stdout") or "").strip()

            chunk = RetrievedChunk(
                source_file="python_sandbox",
                content=stdout,
                relevance_score=1.0,
                retrieval_method="compute",
            )

            context.add_chunk(chunk)

            context.add_claim(
                Claim(
                    text=stdout[:200],
                    source_agent=self.agent_id,
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.95,
                )
            )

    # -----------------------------
    # WEB FALLBACK
    # -----------------------------
    async def _web_search_fallback(self, task, context: SharedContext):

        inp = ToolInput(
            data={"query": task.description, "max_results": MAX_WEB_RESULTS},
            job_id=context.job_id,
            agent_id=self.agent_id,
        )

        try:
            result = await self._web_tool.execute(inp, context)
        except Exception:
            return []

        if not result.success:
            return []

        chunks = []

        for item in result.output.get("results", []):
            snippet = (item.get("snippet") or "").strip()
            if not snippet:
                continue

            chunks.append(
                RetrievedChunk(
                    source_file=item.get("url", "web"),
                    content=snippet,
                    relevance_score=0.7,
                    retrieval_method="web_search",
                )
            )

        return chunks

    # -----------------------------
    # CLAIM EXTRACTION
    # -----------------------------
    async def _extract_claims(self, chunk, context: SharedContext, system: str):

        prompt = f"""
Question: {context.query}

Text:
{chunk.content}
"""

        try:
            raw = await self._call_llm(
                context,
                [{"role": "user", "content": prompt}],
                system=system,
            )
        except Exception:
            return []

        try:
            items = self._parse_json(raw)
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        claims = []

        for item in items[:5]:
            if not isinstance(item, dict):
                continue

            text = (item.get("text") or "").strip()
            if not text:
                continue

            conf = float(item.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))

            claims.append(
                Claim(
                    text=text,
                    source_agent=self.agent_id,
                    source_chunk_id=chunk.chunk_id,
                    confidence=conf,
                )
            )

        return claims