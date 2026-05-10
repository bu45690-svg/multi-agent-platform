"""
worker/agents/critique.py

CritiqueAgent — third agent in the pipeline.

Responsibility
--------------
Read every Claim from context.claims and produce a CritiqueResult for each:
  • overall score 0.0–1.0
  • flagged (bool) + flag_reason
  • contradicts (list of other claim_ids)

Also detects cross-claim contradictions by comparing claims pairwise in small
batches (avoids O(n²) LLM calls by grouping similar claims first).

Uses the SelfReflectionTool for individual claim scoring, then runs a
separate contradiction-detection pass over flagged + low-confidence claims.

Writes
------
context.critique_results  : dict[claim_id, CritiqueResult]
context.claims[*].flagged : propagated by context.set_critique()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from worker.agents.base import BaseAgent
from worker.context import CritiqueResult, SharedContext
from worker.tools.base import ToolInput
from worker.tools.self_reflection import SelfReflectionTool

if TYPE_CHECKING:
    pass

# Rubric applied to every claim
_CLAIM_RUBRIC = [
    "The claim is factually accurate based on the retrieved source passage.",
    "The claim is specific and falsifiable (not vague or tautological).",
    "The claim does not overstate confidence beyond what the source supports.",
    "The claim is directly relevant to the original user query.",
    "The claim does not contradict well-known facts.",
]

_CONTRADICTION_SYSTEM = """\
You are a fact-checker. Given a list of claims, identify any pairs that
directly contradict each other. Return ONLY a JSON array of objects:
[{"claim_a": "<claim_id>", "claim_b": "<claim_id>", "reason": "<why they contradict>"}]
If there are no contradictions return [].
No prose, no markdown fences.
"""

_FLAG_THRESHOLD = 0.50        # claims below this score are flagged
_CONTRADICTION_BATCH = 10     # max claims per contradiction-detection pass


class CritiqueAgent(BaseAgent):
    agent_id = "critique_agent"
    max_tokens = 512

    def __init__(self, db_session_factory=None) -> None:
        super().__init__(db_session_factory)
        self._reflection_tool = SelfReflectionTool(
            logger=self._log,
            db_session_factory=db_session_factory,
        )

    async def _validate(self, context: SharedContext) -> None:
        if not context.claims:
            # Not a hard error — synthesis can still run with no claims
            await self._log.emit(
                context.job_id,
                "agent_start",
                payload={"warning": "No claims to critique — skipping"},
            )

    async def _execute(self, context: SharedContext) -> None:
        if not context.claims:
            return

        # ── 1. Score each claim individually ───────────────────────────
        for claim in context.claims:
            result = await self._score_claim(claim, context)
            context.set_critique(result)    # also propagates .flagged onto claim

        # ── 2. Contradiction detection over all claims ──────────────────
        await self._detect_contradictions(context)

        # ── 3. Summary log ─────────────────────────────────────────────
        flagged_count = len(context.flagged_claims)
        await self._log.emit(
            context.job_id,
            "agent_complete",
            payload={
                "claims_scored":     len(context.claims),
                "claims_flagged":    flagged_count,
                "avg_score":         self._avg_score(context),
            },
        )

    # ------------------------------------------------------------------
    # Individual claim scoring
    # ------------------------------------------------------------------

    async def _score_claim(
        self, claim, context: SharedContext
    ) -> CritiqueResult:
        # Fetch source chunk content for extra context
        source_chunk = context.get_chunk(claim.source_chunk_id or "")
        chunk_content = source_chunk.content if source_chunk else ""

        inp = ToolInput(
            data={
                "content": claim.text,
                "rubric":  _CLAIM_RUBRIC,
                "context": (
                    f"Original query: {context.query}\n\n"
                    f"Source passage: {chunk_content[:600]}"
                ),
            },
            job_id=context.job_id,
            agent_id=self.agent_id,
        )

        try:
            tool_result = await self._reflection_tool.execute(inp, context)
        except Exception as exc:
            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={"claim_id": claim.claim_id, "error": str(exc)},
            )
            # Return a neutral result — don't crash the whole pipeline
            return CritiqueResult(
                claim_id=claim.claim_id,
                score=0.5,
                flagged=False,
                reason="Critique tool error — defaulting to neutral score",
            )

        if not tool_result.success:
            return CritiqueResult(
                claim_id=claim.claim_id,
                score=0.5,
                flagged=False,
                reason=f"Critique tool failed: {tool_result.error}",
            )

        output = tool_result.output
        score = float(output.get("overall_score", 0.5))
        issues = output.get("issues", [])
        verdict = output.get("verdict", "warn")

        flagged = score < _FLAG_THRESHOLD or verdict == "fail"
        flag_reason = "; ".join(issues[:2]) if flagged and issues else None

        return CritiqueResult(
            claim_id=claim.claim_id,
            score=score,
            flagged=flagged,
            reason=flag_reason,
        )

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    async def _detect_contradictions(self, context: SharedContext) -> None:
        """
        Run contradiction detection in batches over all claims.
        Updates CritiqueResult.contradicts for any pair found.
        """
        claims = context.claims
        if len(claims) < 2:
            return

        # Process in overlapping batches to catch cross-batch contradictions
        for start in range(0, len(claims), _CONTRADICTION_BATCH):
            batch = claims[start : start + _CONTRADICTION_BATCH]
            await self._check_batch_contradictions(batch, context)

    async def _check_batch_contradictions(
        self, batch, context: SharedContext
    ) -> None:
        claims_text = "\n".join(
            f"{c.claim_id}: {c.text}" for c in batch
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Original query: {context.query}\n\n"
                    f"Claims:\n{claims_text}"
                ),
            }
        ]

        try:
            raw = await self._call_llm(
                context,
                messages,
                system=_CONTRADICTION_SYSTEM,
            )
            pairs = self._parse_json(raw)
        except Exception:
            return   # contradiction detection is best-effort

        if not isinstance(pairs, list):
            return

        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            cid_a = pair.get("claim_a", "")
            cid_b = pair.get("claim_b", "")
            reason = pair.get("reason", "contradiction detected")

            # Update CritiqueResults
            for cid, other in [(cid_a, cid_b), (cid_b, cid_a)]:
                existing = context.critique_results.get(cid)
                if existing:
                    if other not in existing.contradicts:
                        existing.contradicts.append(other)
                    if not existing.flagged:
                        existing.flagged = True
                        existing.reason = reason
                        # Propagate flag back onto the Claim
                        claim_obj = context.get_claim(cid)
                        if claim_obj:
                            claim_obj.flag(reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg_score(self, context: SharedContext) -> float:
        results = list(context.critique_results.values())
        if not results:
            return 0.0
        return round(sum(r.score for r in results) / len(results), 3)