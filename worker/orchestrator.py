"""
worker/orchestrator.py
Clean, production-safe orchestrator (FIXED)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from google import genai

from logger.structured import get_logger
from worker.agents.decomposition import DecompositionAgent
from worker.agents.retrieval import RetrievalAgent
from worker.agents.critique import CritiqueAgent
from worker.agents.synthesis import SynthesisAgent
from worker.agents.meta import MetaAgent
from worker.context import RoutingPlan, SharedContext
from worker.token_budget import (
    BudgetExceededError,
    BudgetTracker,
    NoBudgetAssignedError,
)

MAX_RETRIES = 2

_DEFAULT_BUDGETS: dict[str, int] = {
    "decomposition_agent": 1500,
    "retrieval_agent": 4000,
    "critique_agent": 3000,
    "synthesis_agent": 3500,
    "meta_agent": 2000,
}

_DEFAULT_AGENT_ORDER = [
    "decomposition_agent",
    "retrieval_agent",
    "critique_agent",
    "synthesis_agent",
]

_ROUTING_SYSTEM = """
You are an orchestration planner.

Return ONLY JSON:
{
  "agents": ["decomposition_agent","retrieval_agent","critique_agent","synthesis_agent"],
  "budgets": {"agent": 1000},
  "reasoning": "..."
}

RULES:
- ALWAYS include decomposition_agent
- Total budget <= 12000
"""


class EmptyResultError(Exception):
    pass


class Orchestrator:

    def __init__(self, retriever: Any | None = None, db_session_factory: Any | None = None):
        self._retriever = retriever
        self._db = db_session_factory
        self._log = get_logger("orchestrator")

        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")

        self._gemini_client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self._agents = self._build_agents()

    # -----------------------------
    # MAIN ENTRY
    # -----------------------------
    async def run(self, query: str, job_id: str | None = None, sse_queue: asyncio.Queue | None = None):

        job_id = job_id or str(uuid.uuid4())
        context = SharedContext(job_id=job_id, query=query, sse_queue=sse_queue)

        await self._log.emit(job_id, "job_start", payload={"query": query})
        t0 = time.perf_counter()

        try:
            plan = await self._build_routing_plan(context)
            context.routing_plan = plan

            # ✅ HARD FIX: ensure decomposition ALWAYS runs first
            if "decomposition_agent" not in plan.agents:
                plan.agents.insert(0, "decomposition_agent")
            elif plan.agents[0] != "decomposition_agent":
                plan.agents.remove("decomposition_agent")
                plan.agents.insert(0, "decomposition_agent")

            await self._log.routing_decision(
                job_id,
                agents=plan.agents,
                budgets=plan.budgets,
                reasoning=plan.reasoning,
            )

            # Execute pipeline
            for agent_id in plan.agents:
                agent = self._agents.get(agent_id)
                if not agent:
                    continue

                await self._run_with_retry(agent, context)

            # Meta agent
            try:
                await self._agents["meta_agent"].run(context)
            except Exception as e:
                await self._log.emit(job_id, "agent_error", payload={"meta_error": str(e)})

        except Exception as e:
            await self._log.emit(job_id, "job_error", payload={"error": str(e)})
            raise

        finally:
            await self._persist_context(context)

            elapsed = int((time.perf_counter() - t0) * 1000)

            await self._log.emit(
                job_id,
                "job_complete",
                latency_ms=elapsed,
                payload={
                    "tokens": context.total_tokens,
                    "answer": context.final_answer is not None,
                },
            )

        return context

    # -----------------------------
    # ROUTING
    # -----------------------------
    async def _build_routing_plan(self, context: SharedContext) -> RoutingPlan:
        try:
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[_ROUTING_SYSTEM, context.query],
                config={"response_mime_type": "application/json"},
            )

            data = _extract_json(response.text)

            return RoutingPlan(
                agents=data.get("agents", _DEFAULT_AGENT_ORDER),
                budgets=data.get("budgets", dict(_DEFAULT_BUDGETS)),
                reasoning=data.get("reasoning", ""),
            )

        except Exception:
            return RoutingPlan(
                agents=_DEFAULT_AGENT_ORDER,
                budgets=dict(_DEFAULT_BUDGETS),
                reasoning="fallback routing",
            )

    # -----------------------------
    # EXECUTION
    # -----------------------------
    async def _run_with_retry(self, agent, context: SharedContext):

        for attempt in range(1, MAX_RETRIES + 2):

            try:
                await agent.run(context)
                return

            except BudgetExceededError:
                context.flag_violation(f"budget_exceeded:{agent.agent_id}")
                return

            except NoBudgetAssignedError:
                return

            except Exception as e:
                if attempt > MAX_RETRIES:
                    await self._log.emit(
                        context.job_id,
                        "agent_error",
                        payload={"agent": agent.agent_id, "error": str(e)},
                    )
                    return

                await self._log.retry(context.job_id, attempt=attempt, reason=str(e))
                await asyncio.sleep(0.5)

    # -----------------------------
    # AGENTS
    # -----------------------------
    def _build_agents(self):
        return {
            "decomposition_agent": DecompositionAgent(self._db),
            "retrieval_agent": RetrievalAgent(self._retriever, self._db),
            "critique_agent": CritiqueAgent(self._db),
            "synthesis_agent": SynthesisAgent(self._db),
            "meta_agent": MetaAgent(self._db),
        }

    # -----------------------------
    # PERSISTENCE (FIXED: datetime safe)
    # -----------------------------
    async def _persist_context(self, context: SharedContext):
        if not self._db:
            return

        try:
            from sqlalchemy import text

            session = self._db()

            snapshot = json.dumps(context.to_snapshot(), default=str)

            await session.execute(
                text("""
                    INSERT INTO logs (job_id, agent_id, event_type, payload, timestamp)
                    VALUES (:job_id, 'orchestrator', 'job_complete', :payload::jsonb, NOW())
                """),
                {
                    "job_id": context.job_id,
                    "payload": snapshot,
                },
            )

            await session.commit()

        except Exception as e:
            await self._log.emit(context.job_id, "persist_error", payload={"error": str(e)})


# -----------------------------
# JSON extractor
# -----------------------------
def _extract_json(raw: str) -> dict:
    import re

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("No JSON found")

    return json.loads(m.group())