"""
worker/tools/self_reflection.py

Self Reflection Tool
Uses Gemini to critique agent outputs against a rubric.

NO anthropic dependency.
Uses google-genai SDK only.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from google import genai

from worker.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolInputError,
    ToolResult,
)

# -------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a rigorous evaluator.

You will receive:
1. Content to evaluate
2. A rubric

Return STRICT JSON only.

Format:
{
  "overall_score": 0.0,
  "criterion_scores": {
    "criterion": 0.0
  },
  "issues": [],
  "suggestions": [],
  "verdict": "pass",
  "reasoning": ""
}

Scoring:
0.0-0.4 = fail
0.4-0.7 = warn
0.7-1.0 = pass

Return ONLY valid JSON.
"""

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """
    Extract JSON from model output.
    """

    match = _JSON_BLOCK_RE.search(raw)

    if match:
        return match.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1:
        return raw[start:end + 1]

    return raw.strip()


def _derive_verdict(score: float) -> str:
    if score >= 0.7:
        return "pass"

    if score >= 0.4:
        return "warn"

    return "fail"


def _validate_structure(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure output structure is always valid.
    """

    score = float(data.get("overall_score", 0.0))
    score = max(0.0, min(1.0, score))

    return {
        "overall_score": score,
        "criterion_scores": data.get(
            "criterion_scores",
            {},
        ),
        "issues": data.get(
            "issues",
            [],
        ),
        "suggestions": data.get(
            "suggestions",
            [],
        ),
        "verdict": data.get(
            "verdict",
            _derive_verdict(score),
        ),
        "reasoning": data.get(
            "reasoning",
            "",
        ),
    }


# -------------------------------------------------------------------
# Tool
# -------------------------------------------------------------------

class SelfReflectionTool(BaseTool):

    name = "self_reflection"

    description = (
        "Critique content against a rubric using Gemini."
    )

    timeout_s = 30.0

    async def _validate_input(
        self,
        inp: ToolInput,
    ) -> None:

        inp.require("content", "rubric")

        content = inp.data.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ToolInputError(
                self.name,
                "content must be a non-empty string",
            )

        rubric = inp.data.get("rubric")

        if (
            not isinstance(rubric, list)
            or len(rubric) == 0
            or not all(
                isinstance(r, str) and r.strip()
                for r in rubric
            )
        ):
            raise ToolInputError(
                self.name,
                "rubric must be a non-empty list of strings",
            )

    async def _run(
        self,
        inp: ToolInput,
    ) -> ToolResult:

        content: str = inp.data["content"]
        rubric: list[str] = inp.data["rubric"]
        context_text: str = inp.data.get(
            "context",
            "",
        )

        rubric_text = "\n".join(
            f"- {item}" for item in rubric
        )

        prompt_parts = []

        if context_text:
            prompt_parts.append(
                f"## Context\n{context_text}"
            )

        prompt_parts.append(
            f"## Content\n{content}"
        )

        prompt_parts.append(
            f"## Rubric\n{rubric_text}"
        )

        user_prompt = "\n\n".join(prompt_parts)

        full_prompt = (
            f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        )

        # -----------------------------------------------------------
        # Gemini Call
        # -----------------------------------------------------------

        def _call_gemini():

            api_key = os.environ.get(
                "GEMINI_API_KEY",
                "",
            )

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set"
                )

            client = genai.Client(
                api_key=api_key
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config={
                    "response_mime_type": "application/json"
                },
            )

            return response.text or ""

        try:

            raw_text = await asyncio.to_thread(
                _call_gemini
            )

        except Exception as exc:

            raise ToolExecutionError(
                self.name,
                f"Gemini API error: {exc}",
            ) from exc

        # -----------------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------------

        json_str = _extract_json(raw_text)

        try:

            parsed = json.loads(json_str)

        except json.JSONDecodeError as exc:

            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid JSON from Gemini: {exc}",
                metadata={
                    "raw_response": raw_text[:500],
                },
            )

        validated = _validate_structure(parsed)

        return ToolResult(
            success=True,
            output=validated,
            metadata={
                "model": "gemini-2.5-flash",
                "rubric_items": len(rubric),
            },
        )