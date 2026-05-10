"""
worker/agents/meta.py

MetaAgent — Gemini-powered self-improving loop.

Uses Google Gemini instead of Anthropic Claude.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from worker.agents.base import BaseAgent
from worker.context import SharedContext
from worker.tools.base import ToolInput
from worker.tools.self_reflection import SelfReflectionTool

if TYPE_CHECKING:
    pass


SCORE_THRESHOLD = 0.65
MAX_REWRITES_PER_JOB = 3


_REWRITE_SYSTEM = """
You are a senior AI prompt engineer.

You will receive:
1. The current system prompt for an AI agent
2. Failure diagnoses
3. Examples of poor outputs

Your task:
Write an improved replacement system prompt.

Rules:
- Keep the same purpose and output structure
- Fix the failure modes specifically
- Add concrete operational instructions
- Avoid vague wording
- Keep the prompt concise
- Output ONLY the final improved prompt
"""


_PROMPT_EVAL_RUBRIC = [
    "The prompt clearly defines the role.",
    "The prompt clearly defines the output format.",
    "The prompt contains concrete instructions.",
    "The prompt handles edge cases correctly.",
    "The prompt prevents the observed failures.",
]


class MetaAgent(BaseAgent):

    agent_id = "meta_agent"
    max_tokens = 2048

    def __init__(self, db_session_factory=None) -> None:
        super().__init__(db_session_factory)

        self._reflection_tool = SelfReflectionTool(
            logger=self._log,
            db_session_factory=db_session_factory,
        )

        # Gemini client
        self._client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def _validate(self, context: SharedContext) -> None:
        pass

    async def _execute(self, context: SharedContext) -> None:

        diagnosis = self._diagnose(context)

        if not diagnosis["underperforming_agents"]:

            await self._log.emit(
                context.job_id,
                "agent_complete",
                payload={
                    "meta_agent": "no rewrites needed",
                    "diagnosis": diagnosis,
                },
            )
            return

        rewrites_proposed = 0

        for agent_id, reasons in diagnosis["underperforming_agents"].items():

            if rewrites_proposed >= MAX_REWRITES_PER_JOB:
                break

            success = await self._propose_rewrite(
                agent_id=agent_id,
                reasons=reasons,
                context=context,
            )

            if success:
                rewrites_proposed += 1

        await self._log.emit(
            context.job_id,
            "routing_decision",
            payload={
                "meta_agent": "self_improvement",
                "rewrites_proposed": rewrites_proposed,
                "diagnosis": diagnosis,
            },
        )

    # ==========================================================
    # Diagnosis
    # ==========================================================

    def _diagnose(self, context: SharedContext) -> dict:

        underperforming: dict[str, list[str]] = {}

        def _flag(agent: str, reason: str):
            underperforming.setdefault(agent, []).append(reason)

        # ------------------------------------------------------
        # Critique score analysis
        # ------------------------------------------------------

        scores = [
            r.score
            for r in context.critique_results.values()
        ]

        avg_score = sum(scores) / len(scores) if scores else 1.0

        if avg_score < SCORE_THRESHOLD:

            for claim in context.claims:

                result = context.critique_results.get(
                    claim.claim_id
                )

                if result and result.score < SCORE_THRESHOLD:

                    _flag(
                        claim.source_agent,
                        f"Low critique score {result.score:.2f}: {claim.text[:120]}"
                    )

        # ------------------------------------------------------
        # Final answer missing
        # ------------------------------------------------------

        if context.final_answer is None:

            _flag(
                "synthesis_agent",
                "Final answer was not generated"
            )

        # ------------------------------------------------------
        # Budget violations
        # ------------------------------------------------------

        budget_violations = [
            v for v in context.policy_violations
            if "budget_exceeded" in v
        ]

        for violation in budget_violations:

            m = re.search(r"agent=(\w+)", violation)

            if m:

                _flag(
                    m.group(1),
                    f"Token budget exceeded: {violation}"
                )

        # ------------------------------------------------------
        # Policy violations
        # ------------------------------------------------------

        policy_violations = [
            v for v in context.policy_violations
            if "policy:" in v
        ]

        for violation in policy_violations:

            _flag(
                "synthesis_agent",
                f"Policy violation: {violation}"
            )

        # ------------------------------------------------------
        # Empty DAG
        # ------------------------------------------------------

        if not context.task_dag:

            _flag(
                "decomposition_agent",
                "Task DAG is empty"
            )

        # ------------------------------------------------------
        # Retrieval failure
        # ------------------------------------------------------

        if not context.retrieved_chunks:

            _flag(
                "retrieval_agent",
                "Retrieved zero chunks"
            )

        return {
            "underperforming_agents": underperforming,
            "avg_critique_score": round(avg_score, 3),
            "budget_violations": budget_violations,
            "policy_violations": policy_violations,
        }

    # ==========================================================
    # Rewrite Proposal
    # ==========================================================

    async def _propose_rewrite(
        self,
        agent_id: str,
        reasons: list[str],
        context: SharedContext,
    ) -> bool:

        current_prompt = await self._load_agent_prompt(agent_id)

        # ------------------------------------------------------
        # Evaluate current prompt quality
        # ------------------------------------------------------

        eval_input = ToolInput(
            data={
                "content": current_prompt,
                "rubric": _PROMPT_EVAL_RUBRIC,
                "context":
                    f"Agent: {agent_id}\n"
                    f"Reasons:\n" + "\n".join(reasons),
            },
            job_id=context.job_id,
            agent_id=self.agent_id,
        )

        try:

            eval_result = await self._reflection_tool.execute(
                eval_input,
                context,
            )

        except Exception as exc:

            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={
                    "meta_eval_error": str(exc),
                    "target_agent": agent_id,
                },
            )

            return False

        if not eval_result.success:
            return False

        prompt_score = eval_result.output.get(
            "overall_score",
            0.5,
        )

        # Skip if prompt already good
        if prompt_score >= 0.75:

            await self._log.emit(
                context.job_id,
                "agent_complete",
                payload={
                    "meta_skip": agent_id,
                    "reason":
                        f"Prompt already high quality "
                        f"({prompt_score:.2f})",
                },
            )

            return False

        # ------------------------------------------------------
        # Build rewrite prompt
        # ------------------------------------------------------

        diagnosis_text = "\n".join(
            f"- {r}" for r in reasons
        )

        bad_outputs = self._collect_bad_outputs(
            agent_id,
            context,
        )

        user_prompt = f"""
## Current Prompt
{current_prompt}

## Failure Diagnosis
{diagnosis_text}

## Examples of Bad Outputs
{bad_outputs}

## Prompt Evaluation Score
{prompt_score:.2f}/1.0

## Task
Rewrite the prompt to solve these failures.
"""

        # ------------------------------------------------------
        # Generate rewrite with Gemini
        # ------------------------------------------------------

        try:

            proposed_prompt = await self._call_gemini(
                system_prompt=_REWRITE_SYSTEM,
                user_prompt=user_prompt,
            )

        except Exception as exc:

            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={
                    "meta_rewrite_error": str(exc),
                    "target_agent": agent_id,
                },
            )

            return False

        proposed_prompt = proposed_prompt.strip()

        if (
            not proposed_prompt
            or proposed_prompt == current_prompt
        ):
            return False

        # ------------------------------------------------------
        # Unified diff
        # ------------------------------------------------------

        diff = "\n".join(
            difflib.unified_diff(
                current_prompt.splitlines(),
                proposed_prompt.splitlines(),
                fromfile=f"{agent_id}/current",
                tofile=f"{agent_id}/proposed",
                lineterm="",
            )
        )

        justification = (
            f"Prompt rewrite proposed for {agent_id}. "
            f"Reasons: {diagnosis_text}"
        )

        # ------------------------------------------------------
        # Persist
        # ------------------------------------------------------

        await self._persist_rewrite(
            agent_id=agent_id,
            current_prompt=current_prompt,
            proposed_prompt=proposed_prompt,
            justification=justification,
            diff=diff,
        )

        await self._log.emit(
            context.job_id,
            "routing_decision",
            payload={
                "meta_rewrite_proposed": agent_id,
                "prompt_score_before": prompt_score,
            },
        )

        return True

    # ==========================================================
    # Gemini
    # ==========================================================

    async def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self._client.models.generate_content(
            model="gemini-2.5-flash",

            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=2048,
            ),

            contents=user_prompt,
        )

        return response.text

    # ==========================================================
    # Helpers
    # ==========================================================

    async def _load_agent_prompt(
        self,
        agent_id: str,
    ) -> str:

        if self._db is None:
            return f"[Default prompt for {agent_id}]"

        try:

            from sqlalchemy import text

            session = self._db()

            row = await session.execute(
                text(
                    """
                    SELECT content
                    FROM prompts
                    WHERE agent_id = :aid
                    AND is_active = TRUE
                    ORDER BY version DESC
                    LIMIT 1
                    """
                ),
                {"aid": agent_id},
            )

            result = row.fetchone()

            return (
                result[0]
                if result
                else f"[No active prompt for {agent_id}]"
            )

        except Exception:
            return f"[DB error loading prompt for {agent_id}]"

    def _collect_bad_outputs(
        self,
        agent_id: str,
        context: SharedContext,
    ) -> str:

        examples: list[str] = []

        if agent_id == "decomposition_agent":

            failed = [
                t for t in context.task_dag
                if t.status == "failed"
            ]

            for t in failed[:2]:

                examples.append(
                    f"Failed task: {t.task_id} -> {t.output}"
                )

        elif agent_id == "retrieval_agent":

            low = [
                c for c in context.claims
                if (
                    context.critique_results.get(
                        c.claim_id
                    )
                    and
                    context.critique_results[
                        c.claim_id
                    ].score < SCORE_THRESHOLD
                )
            ]

            for c in low[:3]:

                examples.append(
                    f"Low-score claim: {c.text}"
                )

        elif agent_id == "synthesis_agent":

            if context.final_answer:

                examples.append(
                    context.final_answer[:500]
                )

            else:

                examples.append(
                    "No final answer generated"
                )

        return (
            "\n".join(examples)
            if examples
            else "(no examples)"
        )

    async def _persist_rewrite(
        self,
        agent_id: str,
        current_prompt: str,
        proposed_prompt: str,
        justification: str,
        diff: str,
    ) -> None:

        if self._db is None:
            return

        try:

            from sqlalchemy import text

            session = self._db()

            row = await session.execute(
                text(
                    """
                    SELECT id
                    FROM prompts
                    WHERE agent_id = :aid
                    AND is_active = TRUE
                    ORDER BY version DESC
                    LIMIT 1
                    """
                ),
                {"aid": agent_id},
            )

            result = row.fetchone()

            original_id = (
                str(result[0])
                if result
                else None
            )

            await session.execute(
                text(
                    """
                    INSERT INTO prompt_rewrites
                    (
                        original_id,
                        proposed_content,
                        justification,
                        diff,
                        status
                    )
                    VALUES
                    (
                        :original_id,
                        :proposed_content,
                        :justification,
                        :diff,
                        'pending'
                    )
                    """
                ),
                {
                    "original_id": original_id,
                    "proposed_content": proposed_prompt,
                    "justification": justification,
                    "diff": diff,
                },
            )

            await session.commit()

        except Exception as exc:

            await self._log.emit(
                "system",
                "agent_error",
                payload={
                    "meta_persist_error": str(exc)
                },
            )