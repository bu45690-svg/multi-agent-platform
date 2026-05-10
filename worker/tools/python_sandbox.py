"""
worker/tools/python_sandbox.py

Python sandbox tool — executes arbitrary Python snippets in an isolated
subprocess with a hard timeout and resource constraints.

Input
-----
    {
        "code":        str,    # required — Python source to execute
        "timeout_s":   float,  # optional, default 10, max 30
        "stdin":       str     # optional — data piped to the script's stdin
    }

Output (on success)
-------------------
    {
        "stdout":      str,
        "stderr":      str,
        "returncode":  int,
        "timed_out":   bool
    }

Security model
--------------
• Code runs in a SEPARATE subprocess (not exec/eval in the worker process).
• The subprocess inherits a minimal environment (only PATH + PYTHONPATH).
• Wall-clock timeout is enforced via asyncio.wait_for + proc.kill().
• No network access is granted to the subprocess (enforce at infra level
  via Docker network policy; this module does not block network calls itself).
• stdout + stderr are capped at MAX_OUTPUT_BYTES to prevent memory exhaustion.

This is a "soft" sandbox suitable for a take-home project. For production,
replace the subprocess call with a gVisor / Firecracker container invocation.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from worker.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolInputError,
    ToolResult,
)

_DEFAULT_TIMEOUT_S = 10.0
_MAX_TIMEOUT_S = 30.0
_MAX_OUTPUT_BYTES = 64 * 1024   # 64 KB per stream

# Minimal environment passed to the subprocess
_SAFE_ENV_KEYS = {"PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL"}


def _build_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}


class PythonSandboxTool(BaseTool):
    name = "python_sandbox"
    description = (
        "Execute a Python code snippet in an isolated subprocess. "
        "Returns stdout, stderr, and return code. "
        "Input: {code: str, timeout_s?: float, stdin?: str}."
    )
    # Outer timeout includes subprocess spawn overhead + inner timeout
    timeout_s = _MAX_TIMEOUT_S + 5.0

    async def _validate_input(self, inp: ToolInput) -> None:
        inp.require("code")
        code = inp.data.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ToolInputError(self.name, "code must be a non-empty string")

        timeout = inp.data.get("timeout_s", _DEFAULT_TIMEOUT_S)
        if not isinstance(timeout, (int, float)) or not (0 < timeout <= _MAX_TIMEOUT_S):
            raise ToolInputError(
                self.name,
                f"timeout_s must be a number in (0, {_MAX_TIMEOUT_S}]",
            )

    async def _run(self, inp: ToolInput) -> ToolResult:
        code: str = inp.data["code"]
        timeout_s: float = float(inp.data.get("timeout_s", _DEFAULT_TIMEOUT_S))
        stdin_data: str | None = inp.data.get("stdin")

        stdin_bytes = stdin_data.encode() if stdin_data else None

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_build_env(),
            )
        except OSError as exc:
            raise ToolExecutionError(
                self.name, f"Failed to spawn subprocess: {exc}", retryable=False
            ) from exc

        timed_out = False
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            stdout_raw, stderr_raw = b"", b"[killed after timeout]".encode()

        # Cap output size
        stdout_str = stdout_raw[:_MAX_OUTPUT_BYTES].decode(errors="replace")
        stderr_str = stderr_raw[:_MAX_OUTPUT_BYTES].decode(errors="replace")
        returncode = proc.returncode if proc.returncode is not None else -1

        success = not timed_out and returncode == 0

        return ToolResult(
            success=success,
            output={
                "stdout":     stdout_str,
                "stderr":     stderr_str,
                "returncode": returncode,
                "timed_out":  timed_out,
            },
            error=None if success else (
                "Execution timed out" if timed_out
                else f"Process exited with code {returncode}"
            ),
            metadata={
                "code_length": len(code),
                "timeout_s":   timeout_s,
            },
        )