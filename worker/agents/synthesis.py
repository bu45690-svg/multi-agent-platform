"""
worker/agents/synthesis.py

SynthesisAgent — fourth and final content agent in the pipeline.

Responsibility
--------------
1. Read task_dag, retrieved_chunks, claims, and critique_results from context.
2. Build a grounded prompt that includes only high-quality (non-flagged or
   resolved) evidence.
3. Call the LLM to produce the final answer, streaming tokens via SSE.
4. Parse per-sentence provenance from the response.
5. Write context.final_answer and context.provenance_map.

Contradiction resolution
------------------------
Flagged claims are included with a "DISPUTED:" prefix so the model can
explicitly acknowledge uncertainty rather than silently dropping evidence.

Budget safety
-------------
The retrieved chunks are truncated to fit within the agent's remaining token
budget before building the prompt (via BudgetTracker.truncate_to_budget).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from worker.agents.base import BaseAgent
from worker.context import ProvenanceEntry, SharedContext
from worker.token_budget import BudgetTracker

if TYPE_CHECKING:
    pass


_SYSTEM_PROMPT = """\
You are a precise and honest synthesis agent. Your job is to answer the user's
question using ONLY the evidence provided. Do not add information from your
training data unless you explicitly label it as background knowledge.

Guidelines
----------
• Answer in clear prose, 2-5 paragraphs.
• If evidence is marked DISPUTED, acknowledge the uncertainty explicitly.
• Cite sources inline as [chunk_id] after each factual sentence.
• If the evidence is insufficient, say so honestly.
• Do not speculate or hallucinate. Confidence should match evidence quality.

After your answer, output a provenance section:
---PROVENANCE---
<sentence excerpt (first 8 words)> | <source_agent> | <chunk_id or "none">
...one line per major factual sentence in your answer...
---END---
"""

_PROVENANCE_BLOCK_RE = re.compile(
    r"---PROVENANCE---\s*(.*?)\s*---END---",
    re.DOTALL,
)
_PROVENANCE_LINE_RE = re.compile(
    r"^(.+?)\s*\|\s*(\w+)\s*\|\s*(\S+)\s*$"
)


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis_agent"
    max_tokens = 1024

    def _default_prompt(self) -> str:
        return _SYSTEM_PROMPT

    async def _validate(self, context: SharedContext) -> None:
        if not context.retrieved_chunks and not context.claims:
            raise ValueError(
                "SynthesisAgent: no chunks or claims in context — "
                "retrieval must run before synthesis"
            )

    async def _execute(self, context: SharedContext) -> None:
        system = await self._load_prompt()
        tracker = BudgetTracker(context)

        # ── 1. Build evidence block ─────────────────────────────────────
        evidence = self._build_evidence(context, tracker)

        # ── 2. Build task summary ───────────────────────────────────────
        task_summary = self._build_task_summary(context)

        # ── 3. Compose prompt ───────────────────────────────────────────
        user_content = (
            f"## User query\n{context.query}\n\n"
            f"## Tasks completed\n{task_summary}\n\n"
            f"## Evidence\n{evidence}"
        )

        messages = [{"role": "user", "content": user_content}]

        # ── 4. Call LLM (streaming) ─────────────────────────────────────
        raw = await self._call_llm(
            context,
            messages,
            system=system,
            stream=True,   # tokens flow to SSE queue
        )

        # ── 5. Parse provenance + strip it from the answer ──────────────
        final_answer, provenance = self._split_provenance(raw, context)
        context.final_answer = final_answer.strip()

        for entry in provenance:
            context.provenance_map.append(entry)

        # Close the SSE stream
        await context.close_sse()

        await self._log.emit(
            context.job_id,
            "agent_complete",
            payload={
                "answer_length":      len(context.final_answer),
                "provenance_entries": len(context.provenance_map),
                "flagged_claims_used": sum(
                    1 for c in context.claims if c.flagged
                ),
            },
        )

    # ------------------------------------------------------------------
    # Evidence builder
    # ------------------------------------------------------------------

    def _build_evidence(self, context: SharedContext, tracker: BudgetTracker) -> str:
        """
        Produce a structured evidence block from chunks + critique results.
        High-quality claims come first; flagged claims are prefixed DISPUTED.
        Truncated to fit within the remaining budget.
        """
        lines: list[str] = []

        # Group by chunk
        chunk_map: dict[str, list] = {}
        for claim in context.claims:
            cid = claim.source_chunk_id or "__agent__"
            chunk_map.setdefault(cid, []).append(claim)

        for chunk in context.retrieved_chunks:
            lines.append(f"### [{chunk.chunk_id}] {chunk.source_file} (score={chunk.relevance_score:.2f})")
            lines.append(chunk.content)

            claims_for_chunk = chunk_map.get(chunk.chunk_id, [])
            if claims_for_chunk:
                lines.append("**Claims extracted:**")
                for claim in claims_for_chunk:
                    critique = context.critique_results.get(claim.claim_id)
                    score_str = f"{critique.score:.2f}" if critique else "?"
                    prefix = "DISPUTED: " if claim.flagged else ""
                    lines.append(
                        f"  • [{claim.claim_id}] (score={score_str}) "
                        f"{prefix}{claim.text}"
                    )
            lines.append("")   # blank line between chunks

        # Agent-only claims (no chunk source)
        agent_claims = chunk_map.get("__agent__", [])
        if agent_claims:
            lines.append("### Agent-derived claims (no source chunk)")
            for claim in agent_claims:
                prefix = "DISPUTED: " if claim.flagged else ""
                lines.append(f"  • [{claim.claim_id}] {prefix}{claim.text}")
            lines.append("")

        full_evidence = "\n".join(lines)

        # Truncate if needed
        try:
            truncated = tracker.truncate_to_budget(
                self.agent_id,
                full_evidence,
                reserve_for_completion=self.max_tokens,
            )
            if len(truncated) < len(full_evidence):
                truncated += "\n\n[Evidence truncated to fit token budget]"
            return truncated
        except Exception:
            return full_evidence

    # ------------------------------------------------------------------
    # Task summary
    # ------------------------------------------------------------------

    def _build_task_summary(self, context: SharedContext) -> str:
        lines: list[str] = []
        for task in context.task_dag:
            status_icon = {"done": "✓", "failed": "✗", "running": "…"}.get(
                task.status, "?"
            )
            lines.append(f"{status_icon} [{task.task_id}] {task.description}")
        return "\n".join(lines) if lines else "(no tasks)"

    # ------------------------------------------------------------------
    # Provenance parsing
    # ------------------------------------------------------------------

    def _split_provenance(
        self,
        raw: str,
        context: SharedContext,
    ) -> tuple[str, list[ProvenanceEntry]]:
        """
        Split the raw LLM output into (answer_text, provenance_entries).
        The provenance block is optional — if the model omits it, provenance
        is empty but the answer is still stored.
        """
        prov_match = _PROVENANCE_BLOCK_RE.search(raw)
        if not prov_match:
            return raw, []

        # Answer = everything before the provenance block
        answer = raw[: prov_match.start()].strip()
        prov_block = prov_match.group(1)

        entries: list[ProvenanceEntry] = []
        for line in prov_block.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _PROVENANCE_LINE_RE.match(line)
            if not m:
                continue
            sentence_excerpt, source_agent, chunk_ref = m.groups()
            chunk_id = chunk_ref if chunk_ref.lower() not in ("none", "n/a", "-") else None
            entries.append(
                ProvenanceEntry(
                    sentence=sentence_excerpt.strip(),
                    source_agent=source_agent.strip(),
                    source_chunk_id=chunk_id,
                )
            )

        return answer, entries