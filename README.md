# Multi-Agent Orchestration Platform

A production-grade multi-agent pipeline built on FastAPI, PostgreSQL, FAISS, and the Anthropic API. Agents communicate exclusively through a typed `SharedContext` object, never calling each other directly.

---

## Architecture Overview

```
User HTTP Request (POST /query)
        │
        ▼
┌──────────────┐   SSE stream (tokens)   ┌──────────────┐
│   FastAPI    │◄────────────────────────│  SSE Client  │
│  (api/)      │                         └──────────────┘
└──────┬───────┘
       │  asyncio.Queue (in-process)
       ▼
┌──────────────┐
│  Async       │  One background task per FastAPI process
│  Worker      │  (no Redis required)
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                     │
│  • LLM-generated routing plan + token budgets       │
│  • Typed retry logic per exception class            │
│  • Persists SharedContext snapshot to PostgreSQL    │
└──┬──────────┬──────────┬────────────┬──────────────┘
   │          │          │            │
   ▼          ▼          ▼            ▼
DECOMP     RETRIEVAL  CRITIQUE    SYNTHESIS
   │          │          │            │
   └──────────┴──────────┴────────────┘
                     │
              SharedContext
           (Pydantic model, RAM)
                     │
                     ▼
              ┌────────────┐
              │ PostgreSQL │
              │ logs       │
              │ eval_runs  │
              │ prompts    │
              │ tool_calls │
              └────────────┘
```

---

## Quickstart

### Prerequisites

- Docker + Docker Compose v2
- An Anthropic API key

### 1. Clone and configure

```bash
git clone <repo>
cd multi-agent-platform
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum
```

### 2. Start all services

```bash
docker compose up --build
```

Services started:
| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI + worker |
| `postgres` | 5432 | PostgreSQL 16 |
| `logviewer` | 8080 | Log viewer UI |

### 3. Submit a query

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS and how does it compare to Annoy?"}' 
```

The `-N` flag disables buffering so SSE tokens stream to the terminal in real time.

### 4. View the trace

```bash
# Copy the job_id from the first SSE event: data: [JOB_ID] <uuid>
curl http://localhost:8000/trace/<job_id> | jq .
```

### 5. Run the eval harness

```bash
curl -X POST http://localhost:8000/eval/run
# Results available in ~2 minutes:
curl http://localhost:8000/eval/latest | jq .
```

---

## Project Structure

```
multi-agent-platform/
├── docker-compose.yml
├── .env.example
├── README.md
├── docs/                          # RAG source documents (*.md)
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                    # 5 endpoints + SSE + lifespan
│
├── worker/
│   ├── main.py                    # WorkerState, job queue, startup/shutdown
│   ├── orchestrator.py            # Routing plan, retry logic, context persistence
│   ├── context.py                 # SharedContext Pydantic model
│   ├── token_budget.py            # BudgetTracker, BudgetExceededError
│   │
│   ├── agents/
│   │   ├── base.py                # BaseAgent ABC
│   │   ├── decomposition.py       # Produces task DAG
│   │   ├── retrieval.py           # FAISS + web_search + claim extraction
│   │   ├── critique.py            # Claim scoring + contradiction detection
│   │   ├── synthesis.py           # Final answer + provenance
│   │   └── meta.py                # Self-improving prompt rewrite proposals
│   │
│   ├── tools/
│   │   ├── base.py                # BaseTool, ToolResult, typed exceptions
│   │   ├── web_search.py          # stub / Brave / Serper backends
│   │   ├── python_sandbox.py      # Subprocess code execution
│   │   ├── nl_to_sql.py           # NL → SQL via LLM + safe execution
│   │   └── self_reflection.py     # LLM-based structured critique
│   │
│   └── rag/
│       ├── indexer.py             # FAISS index builder + manifest cache
│       └── retriever.py           # Single/multi-query + RRF + MMR
│
├── db/
│   ├── models.py                  # SQLAlchemy ORM models
│   ├── schemas.py                 # Pydantic schemas
│   └── migrations/
│       └── 001_init.sql
│
├── eval/
│   ├── test_cases.py              # 15 test cases (baseline/ambiguous/adversarial)
│   ├── scoring.py                 # 6 scoring functions
│   └── harness.py                 # Runner + summary table
│
├── logger/
│   └── structured.py              # JSON logger → stdout + PostgreSQL
│
└── logviewer/
    ├── Dockerfile
    └── app.py                     # HTML log table + /api/logs JSON
```

---

## API Reference

### `POST /query`

Submit a natural-language query. Returns a Server-Sent Events stream.

```
POST /query
Content-Type: application/json

{"query": "Explain how FAISS indexing works.", "job_id": null}
```

SSE event types:
```
data: [JOB_ID] <uuid>          # first event — capture this for /trace
data: <token>                  # streamed answer tokens
data: [DONE]                   # stream complete
data: [ERROR] <message>        # pipeline error
```

### `GET /trace/{job_id}`

Replay all log events for a completed job in chronological order.

```json
{
  "job_id": "...",
  "event_count": 42,
  "events": [
    {"timestamp": "...", "agent_id": "decomposition_agent", "event_type": "agent_start", ...},
    ...
  ],
  "tool_calls": [...]
}
```

### `GET /eval/latest`

Aggregated scores from the most recent eval run.

```json
{
  "by_category": [
    {"test_category": "baseline", "avg_correctness": 0.91, ...}
  ],
  "overall": {
    "avg_correctness": 0.78,
    "avg_citation": 0.85,
    ...
  }
}
```

### `POST /eval/run`

Trigger a full 15-case eval run asynchronously.

```json
{"run_id": "<uuid>", "status": "started"}
```

### `POST /prompts/{id}/approve`

Approve a pending MetaAgent prompt rewrite. Activates the new prompt version.

```json
{"rewrite_id": "...", "agent_id": "retrieval_agent", "new_version": 3, "status": "approved"}
```

---

## Configuration

All configuration is via environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required** |
| `POSTGRES_USER` | `agent` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `secret` | PostgreSQL password |
| `POSTGRES_DB` | `agentdb` | Database name |
| `WEB_SEARCH_BACKEND` | `stub` | `stub` \| `brave` \| `serper` |
| `BRAVE_API_KEY` | — | Required if backend=brave |
| `SERPER_API_KEY` | — | Required if backend=serper |
| `EMBED_MODEL` | `local` | `local` \| `voyage-2` |
| `VOYAGE_API_KEY` | — | Required if EMBED_MODEL=voyage-2 |
| `RAG_CHUNK_SIZE` | `400` | Words per RAG chunk |
| `RAG_CHUNK_STRIDE` | `100` | Overlap words between chunks |
| `EVAL_CONCURRENCY` | `3` | Parallel eval test cases |

---

## Adding RAG Documents

Drop any `.md` files into the `docs/` directory and restart the `api` service. The FAISS index rebuilds automatically at startup when the file manifest changes.

```bash
cp my_new_doc.md docs/
docker compose restart api
```

---

## Adding a New Agent

1. Create `worker/agents/my_agent.py`, subclass `BaseAgent`, set `agent_id`.
2. Implement `async _execute(context: SharedContext) → None`.
3. Register it in `Orchestrator._build_agents()`.
4. The LLM routing plan will include it automatically once it's registered.

---

## Adding a New Tool

1. Create `worker/tools/my_tool.py`, subclass `BaseTool`.
2. Set `name`, `description`, `timeout_s`.
3. Implement `async _validate_input()` and `async _run()`.
4. Instantiate it inside the relevant agent's `__init__`.

The tool's calls are automatically logged to the `tool_calls` table and visible in `/trace/{job_id}`.

---

## Evaluation

Run the harness standalone (no Docker required):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m eval.harness
```

Compare two eval runs by querying `eval_runs`:

```sql
SELECT
    a.test_case_id,
    (a.scores->>'correctness')::float - (b.scores->>'correctness')::float AS correctness_delta
FROM eval_runs a
JOIN eval_runs b USING (test_case_id)
WHERE a.created_at > b.created_at
ORDER BY correctness_delta ASC;
```

---

## Self-Improvement Loop

After every job, `MetaAgent` inspects the pipeline's outputs and proposes prompt rewrites when:
- Average critique score < 0.65
- Any agent exceeded its token budget
- Any policy violations were recorded
- The final answer was not produced

Proposals appear in `prompt_rewrites` with `status='pending'`. A human approves them via:

```bash
curl -X POST http://localhost:8000/prompts/<rewrite_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"approved_by": "alice"}'
```

The new prompt is immediately active for all subsequent jobs.

---

## Security Notes

- The Python sandbox runs code in a **child subprocess** with a stripped environment. Network isolation must be enforced at the Docker network layer.
- The NL→SQL tool enforces **SELECT-only** queries via regex before execution, plus a per-connection `statement_timeout`.
- Prompt injection attempts are detected by `BaseAgent._policy_check()` and logged as policy violations. They do not prevent a response but are visible in the trace.

---

## Log Viewer

Open [http://localhost:8080](http://localhost:8080) to browse all log events in a paginated table. Filter by job ID to see the full trace for a specific query.
