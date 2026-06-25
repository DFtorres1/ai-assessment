# Blossom Banking Helper

A production-minded RAG agent that answers login and security questions for Blossom Banking members and support staff. Built with LangGraph, Claude (Anthropic), ChromaDB, and FastAPI.

> **Architecture deep-dive:** [ARCHITECTURE.md](./ARCHITECTURE.md)  
> **Backend spec & TDD test contracts:** [BACKEND_SPEC.md](./BACKEND_SPEC.md)  
> **Architecture diagram:** [docs/architecture.drawio](./docs/architecture.drawio)

---

## Quick start — Docker (recommended)

```bash
# 1. Export your Anthropic API key (the only required secret)
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Start backend + frontend
docker compose up --build

# 3. Open the chat UI
open http://localhost:3000
```

You can ingest documents through the UI, or load the included sample PDFs via the API:

```bash
curl -s -X POST http://localhost:8000/ingest | jq .
```

All other configuration has sensible defaults baked into `docker-compose.yml`. If you prefer a file over a shell export, create `backend/.env` from the example and Compose will pick it up automatically:

```bash
cp backend/.env.example backend/.env
# Set ANTHROPIC_API_KEY=sk-ant-... in that file
```

The first build downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and bakes it into the image. Subsequent starts are fast. The 8 sample PDFs are already in `sample_docs/` — no generation step needed.

---

## Quick start — local (no Docker)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

uvicorn api.main:app --reload
# API at http://localhost:8000  ·  Swagger UI at /docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

### Ingest the knowledge base

With the backend running, trigger ingestion once:

```bash
curl -s -X POST http://localhost:8000/ingest | jq .
```

This reads the PDFs from `sample_docs/` (or the path set in `PDF_DIR`), chunks them into 500-token windows with 50-token overlap, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB. Re-ingesting is idempotent — existing chunks are replaced.

---

## What it does

- Answers login/security questions grounded in eight internal PDFs — no hallucination
- Cites the exact document, page, and section for every answer
- Calls the US Federal Holidays API when a question involves timing or deadlines
- Enforces scope: warmly redirects questions outside login/security
- Guards against jailbreaks and PII leakage before and after LLM calls
- Streams tokens progressively over SSE so users see responses immediately

---

## Implemented bonus items

| Bonus | Status |
|---|---|
| SSE streaming (`GET /chat/stream`) | Done |
| MCP server exposing the holidays tool | Done |
| Session persistence across restarts (SQLite) | Done |
| Testing framework (pytest, httpx, 99.28% coverage, 115 tests) | Done |
| Quality gates (ruff, mypy, bandit, pre-commit) | Done |
| Rate-limit / backoff around LLM and external APIs (tenacity) | Done |
| Config discipline (`.env.example`, env-based secrets) | Done |

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.11 |
| Docker + Docker Compose | any recent |
| Anthropic API key | `claude-haiku-4-5-20251001` + `claude-sonnet-4-6` |

---

## Environment variables

All variables are read from `backend/.env` (copy from `.env.example`).

```
# Required
ANTHROPIC_API_KEY=sk-ant-...

# App
APP_ENV=development
LOG_LEVEL=INFO

# Models
HAIKU_MODEL=claude-haiku-4-5-20251001
SONNET_MODEL=claude-sonnet-4-6

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=blossom_knowledge

# SQLite session store
SQLITE_DB_PATH=./data/sessions.db

# Retrieval & reflexion
RETRIEVAL_ROUTING_THRESHOLD=0.50
REFLEXION_CONFIDENCE_THRESHOLD=0.60
REFLEXION_EMBEDDING_THRESHOLD=0.60
MAX_REFLEXION_ATTEMPTS=2

# Chunking
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=50

# Session history
SESSION_HISTORY_MAX_TURNS=3

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:5173

# PDF source directory (Docker overrides this via docker-compose.yml)
PDF_DIR=../sample_docs
```

---

## API reference

### `POST /chat`

```json
// Request
{
  "session_id": "uuid",
  "message": "I got locked out. Can I unlock myself?",
  "user_type": "member",
  "temperature": 0.2,
  "top_p": 0.9
}

// Response
{
  "answer": "You can reset this yourself — here's how...",
  "citations": [
    {"doc_name": "01_account_lockout_policy", "page": 2, "section": "Self-Service Unlock"}
  ],
  "tool_calls": [],
  "timing_ms": {"total": 1840}
}
```

`user_type` accepts `"member"` or `"staff"`. Staff get admin-level context.

### `GET /chat/stream`

Server-Sent Events for progressive rendering.

```
GET /chat/stream?session_id=uuid&message=...&user_type=member

data: {"type": "tool_start", "tool": "holidays_api", "input": {...}}
data: {"type": "tool_end",   "tool": "holidays_api", "result": {...}, "duration_ms": 310}
data: {"type": "citations",  "citations": [...]}
data: {"type": "token",      "content": "You "}
data: {"type": "token",      "content": "can "}
...
data: {"type": "done",       "tool_calls": [...], "timing_ms": {...}}
```

`tool_start` / `tool_end` events only appear when the holidays tool is invoked.

### `GET /health`

```json
{"status": "ok", "checks": {"db": "ok", "chroma": "ok"}}
```

### `POST /ingest`

Reads PDFs from `PDF_DIR`, chunks, embeds, and upserts into ChromaDB. Returns:

```json
{
  "chunks_indexed": 142,
  "documents_processed": ["01_account_lockout_policy.pdf", "..."],
  "duration_ms": 4200
}
```

---

## Running tests

```bash
cd backend
.venv/bin/pytest                        # full suite with coverage report
.venv/bin/pytest tests/unit/            # unit tests only
.venv/bin/pytest tests/integration/    # integration tests only (LLMs mocked)
```

Current coverage: **99.28%** across 115 tests. The suite runs entirely offline — LLMs and external APIs are mocked via pytest fixtures.

---

## Running the eval script

```bash
# Start the server first
docker compose up -d backend
# or: uvicorn api.main:app

cd backend
python scripts/eval.py
# or target a different host:
python scripts/eval.py --url http://localhost:8000 --timeout 30
```

Reports per-prompt latency, citations, tool calls, and p50/p95/p99 latency. Exits with code 1 if any prompt exceeds the 5 000 ms SLA.

---

## MCP server

The holidays tool is exposed as an MCP server for Claude Desktop or other MCP-compatible clients.

Tools exposed: `get_federal_holidays`, `is_business_day`, `next_business_day`.

### Local (stdio)

```bash
cd backend
python -m mcp_server.holidays_mcp
```

### Docker / deployed (SSE)

`docker compose up` starts an `mcp` service on port `8001` with SSE transport. Test it with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector sse http://localhost:8001/sse
```

Or call the tools directly with curl:

```bash
# 1. Open the SSE connection in the background, note the session URL printed
curl -N http://localhost:8001/sse &

# 2. Call a tool (replace SESSION_URL with the one received above)
curl -s -X POST "$SESSION_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "is_business_day",
      "arguments": {"date_str": "2026-12-25"}
    },
    "id": 1
  }'
```

To connect Claude Desktop to the deployed server, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blossom-holidays": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

---

## Project layout

```
AITechAssessment/
├── sample_docs/                   # 8 knowledge-base PDFs (committed)
│   ├── 01_account_lockout_policy.pdf
│   ├── 02_password_reset_guide.pdf
│   └── ...
├── docs/
│   ├── architecture.md
│   └── architecture.drawio
├── backend/
│   ├── agent/
│   │   ├── graph.py               # LangGraph StateGraph (7 nodes)
│   │   ├── guards/
│   │   │   ├── input_guard.py     # Regex + Haiku: scope, PII, jailbreak
│   │   │   └── output_guard.py    # Embedding grounding + reflexion
│   │   ├── nodes/                 # router, query_expander, retrieve, answer, holidays, fallback
│   │   ├── schemas.py             # Pydantic structured-output schemas
│   │   └── state.py               # AgentState TypedDict
│   ├── api/
│   │   ├── main.py                # FastAPI app + lifespan
│   │   ├── models.py              # Request/response Pydantic models
│   │   └── routes/                # /chat, /chat/stream, /health, /ingest
│   ├── knowledge/
│   │   ├── ingest.py              # PDF → ChromaDB pipeline
│   │   ├── chunker.py             # 500-token chunks, 50-token overlap
│   │   ├── retriever.py           # ChromaDB cosine search
│   │   └── tagger.py              # Keyword → metadata tag mapping
│   ├── services/
│   │   ├── conversation.py        # Stateful conversation orchestration
│   │   ├── sessions.py            # SQLite session persistence (aiosqlite)
│   │   └── retry.py               # tenacity retry wrappers
│   ├── mcp_server/
│   │   └── holidays_mcp.py        # MCP server for the holidays tool
│   ├── scripts/
│   │   └── eval.py                # 10-prompt evaluation harness
│   ├── tests/
│   │   ├── unit/                  # 89 unit tests
│   │   └── integration/           # 26 end-to-end API tests
│   ├── config.py                  # pydantic-settings, env-driven
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── .env.example
│   └── .dockerignore
├── frontend/                      # Vite + React + TypeScript chat UI
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── ARCHITECTURE.md
└── BACKEND_SPEC.md
```

---

## How the agent works

```
User message
    │
    ▼
INPUT GUARD ── regex pre-screen (~0 ms) ──► block if injection token
    │
    └── Haiku call (~150 ms): scope + PII + jailbreak check
                │
        blocked? ──► 400 / SSE rejection stream
                │ no
                ▼
        router_node ── classifies intent (password_reset, mfa_issue, ...)
                │         routes out_of_scope → rejection_node
                ▼
        query_expander_node ── generates 3 retrieval queries
                │
                ▼
        retrieve_node ── 4× parallel ChromaDB cosine search, deduplicated
                │
                ├── intent == holiday_timing? ──► holidays_tool_node (Nager.Date API)
                │
                ├── confidence < threshold? ──► fallback_node (clarifying question)
                │
                ▼
        answer_node ── Haiku (confidence ≥ 0.90) or Sonnet + retrieved context + citations
                │
        OUTPUT GUARD ── embedding grounding check; reflexion loop if confidence low
                │
                ▼
        Final response (streamed via SSE or returned as JSON)
```

**Latency budget (p95 target ≤ 5 s):**

| Step | Budget |
|---|---|
| Input guard (regex + Haiku) | ~150 ms |
| Router + query expander | ~300 ms |
| ChromaDB retrieval (4× parallel) | ~80 ms |
| Answer generation (Haiku/Sonnet, streamed) | ~1.5–3 s |
| Holidays API (conditional) | ~300 ms |
| **Total p95** | **~2–4 s** |

---

## Design decisions

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full rationale. Key choices:

| Decision | Why |
|---|---|
| Anthropic Claude (Haiku + Sonnet) | Best refusal behavior for banking safety; native tool use; streaming |
| LangGraph StateGraph | Conditional routing and tool nodes map 1:1 to the spec; explicit state |
| `all-MiniLM-L6-v2` embeddings (local) | Zero API cost; ~5 ms/query; strong enough for an 8-PDF domain corpus |
| ChromaDB (persistent) | HNSW vector search + SQLite metadata; survives Docker restarts; no infra overhead |
| SQLite + aiosqlite | Session memory across restarts without running a separate service |
| Single Haiku guard call | Three verdicts (scope + PII + jailbreak) in one round-trip; ≤ 200 ms budget |
| Meta-prompting (Haiku vs Sonnet) | Routes high-confidence answers to the faster/cheaper model automatically |
| Active-Prompt (ExampleBank) | Pre-embeds 20 (message, intent) pairs; top-k similarity at query time improves router accuracy |
