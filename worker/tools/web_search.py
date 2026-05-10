"""
worker/tools/web_search.py

Web search tool — production stub with a real, swappable interface.

Input
-----
    {
        "query":      str,          # required — search query
        "max_results": int          # optional, default 5, max 10
    }

Output (on success)
-------------------
    {
        "results": [
            {
                "title":   str,
                "url":     str,
                "snippet": str,
                "rank":    int
            },
            ...
        ],
        "query":        str,
        "total_results": int
    }

Swapping the backend
--------------------
Set env var WEB_SEARCH_BACKEND to one of:
    "stub"      — returns synthetic results (default, no API key needed)
    "brave"     — calls Brave Search API  (needs BRAVE_API_KEY)
    "serper"    — calls Serper.dev API    (needs SERPER_API_KEY)

The Orchestrator never needs to know which backend is active.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from worker.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolInputError,
    ToolResult,
)

_DEFAULT_MAX_RESULTS = 5
_MAX_ALLOWED_RESULTS = 10

# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


async def _stub_search(query: str, max_results: int) -> list[dict]:
    """
    Synthetic results — no API key, deterministic, useful for unit tests
    and local development.
    """
    await asyncio.sleep(0.05)   # simulate network latency
    return [
        {
            "title":   f"Result {i + 1} for: {query}",
            "url":     f"https://example.com/result-{i + 1}",
            "snippet": (
                f"This is a synthetic search result #{i + 1} for the query "
                f"'{query}'. In production this would contain real content."
            ),
            "rank": i + 1,
        }
        for i in range(max_results)
    ]


async def _brave_search(query: str, max_results: int) -> list[dict]:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise ToolExecutionError("web_search", "BRAVE_API_KEY not set", retryable=False)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for i, item in enumerate(data.get("web", {}).get("results", [])[:max_results]):
        results.append(
            {
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("description", ""),
                "rank":    i + 1,
            }
        )
    return results


async def _serper_search(query: str, max_results: int) -> list[dict]:
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        raise ToolExecutionError("web_search", "SERPER_API_KEY not set", retryable=False)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for i, item in enumerate(data.get("organic", [])[:max_results]):
        results.append(
            {
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "rank":    i + 1,
            }
        )
    return results


_BACKENDS: dict[str, Any] = {
    "stub":   _stub_search,
    "brave":  _brave_search,
    "serper": _serper_search,
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current information. "
        "Use when the knowledge base does not cover the query. "
        "Input: {query: str, max_results?: int}."
    )
    timeout_s = 20.0

    async def _validate_input(self, inp: ToolInput) -> None:
        inp.require("query")
        query = inp.data["query"]
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError(self.name, "query must be a non-empty string")

        max_r = inp.data.get("max_results", _DEFAULT_MAX_RESULTS)
        if not isinstance(max_r, int) or not (1 <= max_r <= _MAX_ALLOWED_RESULTS):
            raise ToolInputError(
                self.name,
                f"max_results must be an integer between 1 and {_MAX_ALLOWED_RESULTS}",
            )

    async def _run(self, inp: ToolInput) -> ToolResult:
        query = inp.data["query"].strip()
        max_results = inp.data.get("max_results", _DEFAULT_MAX_RESULTS)

        backend_name = os.environ.get("WEB_SEARCH_BACKEND", "stub").lower()
        backend_fn = _BACKENDS.get(backend_name)
        if backend_fn is None:
            raise ToolExecutionError(
                self.name,
                f"Unknown WEB_SEARCH_BACKEND={backend_name!r}. "
                f"Valid: {list(_BACKENDS)}",
                retryable=False,
            )

        try:
            results = await backend_fn(query, max_results)
        except ToolExecutionError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                self.name,
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(self.name, f"Network error: {exc}") from exc

        return ToolResult(
            success=True,
            output={
                "results":       results,
                "query":         query,
                "total_results": len(results),
            },
            metadata={"backend": backend_name},
        )