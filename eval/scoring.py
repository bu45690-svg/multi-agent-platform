"""
eval/scoring.py

Six scoring functions for the multi-agent eval harness.

Each scorer receives a (TestCase, SharedContext) pair and returns a
ScoreResult with a float score (0.0–1.0) and a text justification.

Scorers
-------
1. correctness          — answer vs ground truth (LLM-as-judge)
2. citation_accuracy    — % claims with valid chunk citations
3. contradiction_resolution — flagged contradictions resolved in final answer
4. tool_efficiency      — penalises unnecessary tool calls
5. context_compliance   — token budget respected across all agents
6. critique_agreement   — synthesis handled critique flags correctly
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from eval.test_cases import TestCase
    from worker.context import SharedContext


@dataclass
class ScoreResult:
    scorer: str
    score: float          # 0.0–1.0
    justification: str
    detail: dict | None = None


# ---------------------------------------------------------------------------
# LLM judge helper (used by correctness scorer)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an objective evaluator. Score how well the candidate answer addresses
the question and matches the ground truth. Return ONLY a JSON object:
{"score": <float 0.0-1.0>, "justification": "<one sentence>"}

Scoring guide:
  1.0 — fully correct, matches ground truth, no hallucinations
  0.7 — mostly correct, minor omissions or imprecision
  0.5 — partially correct, key points missing
  0.3 — largely incorrect but not harmful
  0.0 — wrong, harmful, or refused when it shouldn't have
"""


async def _llm_judge(
    query: str,
    ground_truth: str,
    candidate: str,
) -> tuple[float, str]:
    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    prompt = (
        f"Question: {query}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"Candidate answer: {candidate[:1500]}"
    )
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
        import json
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score = float(data.get("score", 0.5))
        just = str(data.get("justification", ""))
        return max(0.0, min(1.0, score)), just
    except Exception as exc:
        return 0.5, f"Judge error: {exc}"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------


async def score_correctness(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    LLM-as-judge: compare the final answer against ground truth.
    If expect_refusal=True and the answer acknowledges inability, score is 1.0.
    """
    answer = context.final_answer or ""

    if test.expect_refusal:
        # Check whether the answer contains a refusal or strong caveat
        refusal_signals = [
            "cannot", "unclear", "ambiguous", "please clarify",
            "not able", "don't know", "uncertain", "unspecified",
            "refuse", "cannot confirm", "policy",
        ]
        lower = answer.lower()
        if any(s in lower for s in refusal_signals):
            return ScoreResult(
                scorer="correctness",
                score=1.0,
                justification="Correctly refused / caveated as expected.",
            )
        else:
            return ScoreResult(
                scorer="correctness",
                score=0.0,
                justification="Expected refusal/caveat but system gave a direct answer.",
            )

    if not answer:
        return ScoreResult(
            scorer="correctness",
            score=0.0,
            justification="No final answer produced.",
        )

    score, justification = await _llm_judge(test.query, test.ground_truth, answer)
    return ScoreResult(scorer="correctness", score=score, justification=justification)


# ---------------------------------------------------------------------------
# 2. Citation accuracy
# ---------------------------------------------------------------------------


def score_citation_accuracy(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    % of claims that have a valid source_chunk_id pointing to a real chunk.
    If expected_citations=False, returns 1.0 (citations not required).
    """
    if not test.expected_citations:
        return ScoreResult(
            scorer="citation_accuracy",
            score=1.0,
            justification="Citations not required for this test case.",
        )

    claims = context.claims
    if not claims:
        return ScoreResult(
            scorer="citation_accuracy",
            score=0.0,
            justification="No claims extracted — citation accuracy undefined.",
        )

    chunk_ids = {c.chunk_id for c in context.retrieved_chunks}
    valid = sum(
        1 for c in claims
        if c.source_chunk_id and c.source_chunk_id in chunk_ids
    )
    score = valid / len(claims)

    return ScoreResult(
        scorer="citation_accuracy",
        score=round(score, 3),
        justification=f"{valid}/{len(claims)} claims have valid chunk citations.",
        detail={"valid": valid, "total": len(claims)},
    )


# ---------------------------------------------------------------------------
# 3. Contradiction resolution
# ---------------------------------------------------------------------------


def score_contradiction_resolution(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    Of all claims flagged as contradicting, how many are acknowledged in the
    final answer (either resolved or explicitly noted as disputed)?
    """
    contradicted_claims = [
        c for c in context.claims
        if context.critique_results.get(c.claim_id) and
        context.critique_results[c.claim_id].contradicts
    ]

    if not contradicted_claims:
        return ScoreResult(
            scorer="contradiction_resolution",
            score=1.0,
            justification="No contradictions detected — full score.",
        )

    answer = (context.final_answer or "").lower()
    resolved = 0
    for claim in contradicted_claims:
        # Check if the answer acknowledges the claim text or uses hedge language
        key_words = claim.text.lower().split()[:4]
        phrase = " ".join(key_words)
        if phrase in answer or any(w in answer for w in ["disputed", "uncertain", "conflict", "however"]):
            resolved += 1

    score = resolved / len(contradicted_claims)
    return ScoreResult(
        scorer="contradiction_resolution",
        score=round(score, 3),
        justification=(
            f"{resolved}/{len(contradicted_claims)} contradictions acknowledged in answer."
        ),
        detail={"resolved": resolved, "total": len(contradicted_claims)},
    )


# ---------------------------------------------------------------------------
# 4. Tool efficiency
# ---------------------------------------------------------------------------


def score_tool_efficiency(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    Penalises excessive or redundant tool calls.

    Heuristic thresholds:
      ≤ 3 tool calls → 1.0
      4–6           → 0.7
      7–10          → 0.4
      > 10          → 0.1
    """
    # Count tool calls recorded in policy violations or infer from chunks
    # (tool_calls table is in DB; here we use the web_search chunks as proxy)
    web_chunks = [c for c in context.retrieved_chunks if c.retrieval_method == "web_search"]
    compute_chunks = [c for c in context.retrieved_chunks if c.retrieval_method == "sql"]
    total_tool_calls = len(web_chunks) + len(compute_chunks)

    if total_tool_calls <= 3:
        score, note = 1.0, "Efficient tool usage."
    elif total_tool_calls <= 6:
        score, note = 0.7, "Moderate tool usage."
    elif total_tool_calls <= 10:
        score, note = 0.4, "High tool usage — possible redundancy."
    else:
        score, note = 0.1, "Excessive tool calls."

    return ScoreResult(
        scorer="tool_efficiency",
        score=score,
        justification=f"{total_tool_calls} tool call(s). {note}",
        detail={"tool_calls": total_tool_calls},
    )


# ---------------------------------------------------------------------------
# 5. Context compliance (budget)
# ---------------------------------------------------------------------------


def score_context_compliance(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    1.0 if no budget violations; decreasing score per violation.
    """
    budget_violations = [
        v for v in context.policy_violations if "budget_exceeded" in v
    ]

    if not budget_violations:
        return ScoreResult(
            scorer="context_compliance",
            score=1.0,
            justification="All agents stayed within token budget.",
        )

    # Each violation deducts 0.25, floored at 0.0
    score = max(0.0, 1.0 - 0.25 * len(budget_violations))
    return ScoreResult(
        scorer="context_compliance",
        score=round(score, 3),
        justification=f"{len(budget_violations)} budget violation(s) recorded.",
        detail={"violations": budget_violations},
    )


# ---------------------------------------------------------------------------
# 6. Critique agreement
# ---------------------------------------------------------------------------


def score_critique_agreement(
    test: "TestCase",
    context: "SharedContext",
) -> ScoreResult:
    """
    Did SynthesisAgent handle critique flags correctly?

    For each flagged claim, check whether the final answer either:
      (a) omits the claim entirely, OR
      (b) explicitly marks it as disputed/uncertain.

    Score = proportion of flagged claims handled correctly.
    """
    flagged = context.flagged_claims
    if not flagged:
        return ScoreResult(
            scorer="critique_agreement",
            score=1.0,
            justification="No flagged claims — full score.",
        )

    answer = (context.final_answer or "").lower()
    hedge_words = {"disputed", "uncertain", "however", "but", "conflict",
                   "contradict", "unclear", "caveat", "may", "might", "possibly"}

    handled = 0
    for claim in flagged:
        key_words = set(claim.text.lower().split()[:6])
        in_answer = any(w in answer for w in key_words if len(w) > 4)
        hedged = any(h in answer for h in hedge_words)

        if not in_answer:
            # Claim was silently dropped — acceptable if flagged
            handled += 1
        elif hedged:
            # Claim is in the answer but appropriately hedged
            handled += 1
        # else: claim is stated as fact despite being flagged → not handled

    score = handled / len(flagged)
    return ScoreResult(
        scorer="critique_agreement",
        score=round(score, 3),
        justification=(
            f"{handled}/{len(flagged)} flagged claims handled correctly "
            f"(omitted or hedged)."
        ),
        detail={"handled": handled, "total_flagged": len(flagged)},
    )


# ---------------------------------------------------------------------------
# Score all at once
# ---------------------------------------------------------------------------


async def score_all(
    test: "TestCase",
    context: "SharedContext",
) -> dict[str, ScoreResult]:
    """Run all 6 scorers and return a dict keyed by scorer name."""
    correctness = await score_correctness(test, context)
    return {
        "correctness":              correctness,
        "citation_accuracy":        score_citation_accuracy(test, context),
        "contradiction_resolution": score_contradiction_resolution(test, context),
        "tool_efficiency":          score_tool_efficiency(test, context),
        "context_compliance":       score_context_compliance(test, context),
        "critique_agreement":       score_critique_agreement(test, context),
    }


def scores_to_dict(results: dict[str, ScoreResult]) -> dict[str, float]:
    """Flatten to {scorer_name: float} for JSONB storage."""
    return {k: round(v.score, 4) for k, v in results.items()}