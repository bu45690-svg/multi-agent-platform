"""
worker/tools/nl_to_sql.py

NL→SQL tool — translates a natural-language question into a SQL query via the
LLM, then executes it against the application's PostgreSQL database.

Input
-----
    {
        "question":    str,          # required — natural-language question
        "table_hints": list[str]     # optional — table names to scope generation
    }

Output (on success)
-------------------
    {
        "sql":         str,          # generated SQL (SELECT only)
        "rows":        list[dict],   # query results (max MAX_ROWS rows)
        "row_count":   int,
        "column_names": list[str]
    }

Safety constraints
------------------
• Only SELECT statements are executed. Any other statement kind raises
  ToolExecutionError(retryable=False).
• Result rows are capped at MAX_ROWS to prevent runaway result sets.
• A 10-second statement timeout is set per connection.
• The LLM is shown only the *schema* (table + column names), never data.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
import asyncpg

from worker.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolInputError,
    ToolResult,
)

_MAX_ROWS = 100
_STATEMENT_TIMEOUT_MS = 10_000   # 10 s

# ---------------------------------------------------------------------------
# Schema introspection (cached at first call)
# ---------------------------------------------------------------------------

_schema_cache: str | None = None


async def _fetch_schema(dsn: str, table_hints: list[str] | None) -> str:
    """
    Return a compact DDL-like schema string from the live database.
    Only tables listed in table_hints are included (all tables if hints empty).
    """
    global _schema_cache

    async with await asyncpg.connect(dsn) as conn:
        where_clause = ""
        args: list[Any] = []
        if table_hints:
            placeholders = ", ".join(f"${i+1}" for i in range(len(table_hints)))
            where_clause = f"AND c.table_name IN ({placeholders})"
            args = table_hints

        rows = await conn.fetch(
            f"""
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable
            FROM information_schema.columns c
            WHERE c.table_schema = 'public'
              {where_clause}
            ORDER BY c.table_name, c.ordinal_position
            """,
            *args,
        )

    # Build compact schema representation
    tables: dict[str, list[str]] = {}
    for row in rows:
        t = row["table_name"]
        nullable = "" if row["is_nullable"] == "YES" else " NOT NULL"
        tables.setdefault(t, []).append(
            f"  {row['column_name']} {row['data_type']}{nullable}"
        )

    lines = []
    for tname, cols in tables.items():
        lines.append(f"TABLE {tname} (\n" + ",\n".join(cols) + "\n)")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM SQL generation
# ---------------------------------------------------------------------------

_SQL_SYSTEM = """You are a PostgreSQL expert. Given a database schema and a
natural-language question, output ONLY the SQL SELECT query — no explanation,
no markdown fences, no semicolons at the end. The query must be read-only
(SELECT only). If the question cannot be answered with a SELECT query, output
exactly: CANNOT_ANSWER"""

_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_sql(raw: str) -> str:
    """Strip markdown fences if the model added them despite instructions."""
    m = _SQL_BLOCK_RE.search(raw)
    return (m.group(1) if m else raw).strip()


async def _generate_sql(question: str, schema: str) -> str:
    import asyncio
    from google import genai
    import os

    prompt = f"{_SQL_SYSTEM}\n\nSchema:\n{schema}\n\nQuestion: {question}"

    def _call():
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )

    response = await asyncio.to_thread(_call)
    raw = response.text or ""
    m = _BLOCK_RE.search(raw)
    return (m.group(1) if m else raw).strip()


# ---------------------------------------------------------------------------
# SQL safety check
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


def _assert_select_only(sql: str) -> None:
    """Raise ToolExecutionError if sql contains any DML/DDL statement."""
    if _FORBIDDEN_KEYWORDS.search(sql):
        raise ToolExecutionError(
            "nl_to_sql",
            f"Generated SQL contains forbidden statement kind: {sql[:120]}",
            retryable=False,
        )
    if not sql.upper().lstrip().startswith("SELECT"):
        raise ToolExecutionError(
            "nl_to_sql",
            f"Generated SQL does not start with SELECT: {sql[:120]}",
            retryable=False,
        )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class NLToSQLTool(BaseTool):
    name = "nl_to_sql"
    description = (
        "Translate a natural-language question into SQL and execute it "
        "against the application database. "
        "Input: {question: str, table_hints?: list[str]}."
    )
    timeout_s = 25.0

    async def _validate_input(self, inp: ToolInput) -> None:
        inp.require("question")
        q = inp.data.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ToolInputError(self.name, "question must be a non-empty string")

        hints = inp.data.get("table_hints", [])
        if not isinstance(hints, list) or not all(isinstance(h, str) for h in hints):
            raise ToolInputError(self.name, "table_hints must be a list of strings")

    async def _run(self, inp: ToolInput) -> ToolResult:
        question: str = inp.data["question"].strip()
        table_hints: list[str] = inp.data.get("table_hints", [])

        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise ToolExecutionError(
                self.name, "DATABASE_URL env var not set", retryable=False
            )

        # 1. Fetch schema
        try:
            schema = await _fetch_schema(dsn, table_hints or None)
        except Exception as exc:
            raise ToolExecutionError(
                self.name, f"Schema introspection failed: {exc}"
            ) from exc

        if not schema:
            raise ToolExecutionError(
                self.name,
                "No tables found"
                + (f" matching hints {table_hints}" if table_hints else ""),
                retryable=False,
            )

        # 2. Generate SQL
        try:
            sql = await _generate_sql(question, schema)
        except Exception as exc:
            raise ToolExecutionError(
                self.name, f"LLM SQL generation failed: {exc}"
            ) from exc

        if sql == "CANNOT_ANSWER":
            return ToolResult(
                success=False,
                output=None,
                error="LLM determined the question cannot be answered with a SELECT query.",
            )

        # 3. Safety check
        _assert_select_only(sql)

        # 4. Execute
        try:
            async with await asyncpg.connect(dsn) as conn:
                await conn.execute(
                    f"SET statement_timeout = {_STATEMENT_TIMEOUT_MS}"
                )
                rows = await conn.fetch(sql)
        except asyncpg.exceptions.QueryCanceledError:
            raise ToolExecutionError(
                self.name, f"Query timed out after {_STATEMENT_TIMEOUT_MS}ms"
            )
        except asyncpg.exceptions.PostgresError as exc:
            raise ToolExecutionError(
                self.name, f"Query execution error: {exc}"
            ) from exc

        capped = rows[:_MAX_ROWS]
        column_names = list(capped[0].keys()) if capped else []
        result_rows = [dict(r) for r in capped]

        return ToolResult(
            success=True,
            output={
                "sql":          sql,
                "rows":         result_rows,
                "row_count":    len(result_rows),
                "column_names": column_names,
            },
            metadata={
                "total_db_rows": len(rows),
                "capped":        len(rows) > _MAX_ROWS,
            },
        )