## ══════════════════════════════════════════════════════════════════════════
## FILE: logviewer/app.py
## ══════════════════════════════════════════════════════════════════════════
"""
Lightweight log viewer — serves a paginated HTML table of the logs table
and a JSON API at /api/logs.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_engine = None
_session_factory = None


@asynccontextmanager
async def lifespan(app):
    global _engine, _session_factory
    db_url = os.environ["DATABASE_URL"].replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    _engine = create_async_engine(db_url, pool_size=2)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    yield
    await _engine.dispose()


app = FastAPI(title="Log Viewer", lifespan=lifespan)


@app.get("/api/logs")
async def api_logs(
    job_id: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    async with _session_factory() as session:
        where = "WHERE job_id = :job_id" if job_id else ""
        params = {"limit": limit, "offset": offset}
        if job_id:
            params["job_id"] = job_id
        rows = await session.execute(
            text(
                f"""
                SELECT id, timestamp, job_id, agent_id, event_type,
                       latency_ms, token_count, policy_violations, payload
                FROM logs
                {where}
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            params,
        )
        return [dict(r._mapping) for r in rows.fetchall()]


@app.get("/", response_class=HTMLResponse)
async def index(job_id: str | None = Query(None)):
    rows = await api_logs(job_id=job_id, limit=200, offset=0)
    job_filter = f'value="{job_id}"' if job_id else ""

    rows_html = "".join(
        f"<tr>"
        f"<td>{r.get('timestamp','')}</td>"
        f"<td>{r.get('agent_id','')}</td>"
        f"<td><code>{r.get('event_type','')}</code></td>"
        f"<td>{r.get('latency_ms','')}</td>"
        f"<td>{r.get('token_count','')}</td>"
        f"<td style='font-size:11px'>{str(r.get('payload',''))[:120]}</td>"
        f"</tr>"
        for r in rows
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Log Viewer</title>
  <style>
    body {{ font-family: monospace; padding: 16px; background:#111; color:#eee; }}
    table {{ border-collapse: collapse; width:100%; }}
    th,td {{ border:1px solid #333; padding:6px 8px; text-align:left; font-size:12px; }}
    th {{ background:#222; }}
    tr:nth-child(even) {{ background:#1a1a1a; }}
    input {{ background:#222; color:#eee; border:1px solid #444; padding:4px 8px; }}
    button {{ padding:4px 12px; }}
  </style>
</head>
<body>
<h2>🔍 Log Viewer</h2>
<form method="get">
  <label>Job ID: <input name="job_id" size="40" {job_filter}/></label>
  <button type="submit">Filter</button>
  <a href="/" style="margin-left:8px;color:#aaa">Clear</a>
</form>
<p>{len(rows)} events</p>
<table>
  <thead>
    <tr>
      <th>Timestamp</th><th>Agent</th><th>Event</th>
      <th>Latency(ms)</th><th>Tokens</th><th>Payload</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""