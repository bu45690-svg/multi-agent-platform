"""
worker/agents/decomposition.py

DecompositionAgent — first agent in the pipeline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from worker.agents.base import BaseAgent
from worker.context import TaskNode, SharedContext

if TYPE_CHECKING:
    pass


MAX_TASKS = 8


_SYSTEM_PROMPT = """\
You are a task decomposition expert.

Return ONLY JSON array.

Each element:
{
  "task_id": "t1",
  "task_type": "retrieve" | "compute" | "synthesize",
  "description": "...",
  "dependencies": ["t1", "t2"]
}

Rules:
- Always end with a synthesize task
- Keep max 6 tasks
- If ambiguous → single retrieve task
"""


class DecompositionAgent(BaseAgent):

    agent_id = "decomposition_agent"
    max_tokens = 512

    def _default_prompt(self) -> str:
        return _SYSTEM_PROMPT

    async def _validate(self, context: SharedContext) -> None:
        if not context.query.strip():
            raise ValueError("Empty query")

    async def _execute(self, context: SharedContext) -> None:

        system = await self._load_prompt()

        messages = [
            {
                "role": "user",
                "content": f"Decompose:\n\n{context.query}",
            }
        ]

        raw = await self._call_llm(context, messages, system=system)

        try:
            nodes = self._parse_dag(raw, context)

        except Exception as exc:

            await self._log.emit(
                context.job_id,
                "agent_error",
                payload={"error": str(exc), "fallback": True},
            )

            nodes = [
                TaskNode(
                    task_id="t1",
                    task_type="retrieve",
                    description=context.query,
                    dependencies=[],
                )
            ]

        # ================================
        # CRITICAL FIX (YOUR BUG)
        # ================================

        # 1. store in internal system
        for node in nodes:
            context.add_task(node)

        # 2. expose DAG for RetrievalAgent  ⭐ FIX
        context.task_dag = nodes

        await self._log.emit(
            context.job_id,
            "agent_complete",
            payload={
                "task_count": len(nodes),
                "task_types": [n.task_type for n in nodes],
                "dag_set": True,
            },
        )

    # -------------------------------------------------------
    # DAG PARSER
    # -------------------------------------------------------

    def _parse_dag(self, raw: str, context: SharedContext) -> list[TaskNode]:

        data = self._parse_json(raw)

        if not isinstance(data, list):
            raise ValueError("Expected JSON list")

        if len(data) == 0:
            raise ValueError("Empty DAG")

        if len(data) > MAX_TASKS:
            context.flag_violation(
                f"Too many tasks: {len(data)} capped to {MAX_TASKS}"
            )
            data = data[:MAX_TASKS]

        nodes: list[TaskNode] = []
        task_ids = set()

        # ================================
        # FIXED DEPENDENCY LOGIC
        # ================================

        for i, item in enumerate(data):

            if not isinstance(item, dict):
                raise ValueError(f"Invalid node: {item}")

            task_id = item.get("task_id") or f"t{i+1}"
            task_type = item.get("task_type", "retrieve")

            if task_type not in ("retrieve", "compute", "synthesize"):
                task_type = "retrieve"

            description = item.get("description") or f"Task {task_id}"

            # IMPORTANT FIX:
            # add BEFORE dependency filtering so later nodes can depend on earlier ones
            task_ids.add(task_id)

            raw_deps = item.get("dependencies", [])
            deps = [d for d in raw_deps if d in task_ids]

            nodes.append(
                TaskNode(
                    task_id=task_id,
                    task_type=task_type,
                    description=description,
                    dependencies=deps,
                )
            )

        # ensure synthesize at end
        if nodes and nodes[-1].task_type != "synthesize":

            nodes.append(
                TaskNode(
                    task_id=f"t{len(nodes)+1}",
                    task_type="synthesize",
                    description="Synthesize final answer",
                    dependencies=[n.task_id for n in nodes],
                )
            )

        return nodes