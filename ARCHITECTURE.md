# Blossom Banking Helper — Architecture & Technology Decisions

> RAG-based agentic service for login & security Q&A  
> Author: Daniel Torres · dtorres@blossom.technology  
> Date: 2026-06-23
>
> **Detailed backend spec (guardrails, prompts, TDD tests): [BACKEND_SPEC.md](./BACKEND_SPEC.md)**

---

## 1. Technology Stack

### LLM — Anthropic Claude (claude-haiku-4-5 / claude-sonnet-4-6)

| Factor                   | Decision                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Choice**               | Anthropic API (`claude-haiku-4-5` default, `claude-sonnet-4-6` for complex queries)                                                                                      |
| **Why**                  | Best-in-class instruction following, native tool use, streaming SSE, and citations. Haiku hits latency SLA easily; Sonnet available for fallback/complex flows.          |
| **Trade-off vs Bedrock** | Direct Anthropic API is simpler to set up (no IAM config). Bedrock would add enterprise controls and cost visibility — easy swap since LangChain abstracts the provider. |
| **Trade-off vs OpenAI**  | Claude has stronger refusal behavior (safety rails fit the banking scope perfectly) and first-class streaming.                                                           |

### Orchestration — LangGraph

| Factor                          | Decision                                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Choice**                      | `langgraph` with typed `StateGraph`                                                                                                                                                                          |
| **Why**                         | The spec's suggested graph (router → classify → retrieve → answer → holidays_tool → fallback) maps 1:1 to LangGraph nodes/edges. It handles conditional branching, tool nodes, and streaming out of the box. |
| **Trade-off vs LangChain LCEL** | LangGraph gives explicit state management, easier observability per node, and cleaner conditional routing. LCEL is simpler but harder to reason about for multi-step agents with fallbacks.                  |

### Embeddings — `sentence-transformers` (local, `all-MiniLM-L6-v2`)

| Factor                             | Decision                                                                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Choice**                         | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace                                                                                                                   |
| **Why**                            | Zero API cost, runs locally in Docker, fast inference (~5ms/chunk), strong semantic similarity for English banking text. No external dependency or API key.                |
| **Trade-off vs OpenAI embeddings** | OpenAI `text-embedding-3-small` scores marginally higher on benchmarks but adds latency and cost. For a domain-specific corpus of 6 PDFs, local embeddings are sufficient. |

### Vector Store — ChromaDB

| Factor                             | Decision                                                                                                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Choice**                         | `chromadb` with persistent local storage (`./data/chroma`)                                                                                                                    |
| **Why**                            | Persistent by default (survives Docker restarts), excellent LangChain integration, simple metadata filtering (doc_name, page, tag), zero infrastructure overhead.             |
| **Trade-off vs FAISS**             | FAISS is faster at scale but requires manual serialization/deserialization and lacks built-in metadata filtering. For 6 PDFs (~hundreds of chunks), Chroma is the right call. |
| **Trade-off vs Pinecone/Weaviate** | Cloud DBs add network latency and API keys. Out of scope for local-first requirement.                                                                                         |

### API Framework — FastAPI

| Factor     | Decision                                                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Choice** | `fastapi` + `uvicorn`                                                                                                                                                     |
| **Why**    | Native async, SSE via `StreamingResponse`, automatic OpenAPI docs, best-in-class pydantic validation. Meets the `/chat` and `/chat/stream` spec with minimal boilerplate. |

### Session Persistence — SQLite (via `aiosqlite`)

| Factor                 | Decision                                                                                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Choice**             | SQLite with `aiosqlite` (bonus item)                                                                                                                     |
| **Why**                | Zero infrastructure — persists session history across restarts without running a separate service. Trivially swappable to Postgres later via SQLAlchemy. |
| **Trade-off vs Redis** | Redis would add horizontal scalability and TTL support. Overkill for this assessment; noted as the upgrade path.                                         |

### UI — React (Vite) minimal chat page

| Factor     | Decision                                                                             |
| ---------- | ------------------------------------------------------------------------------------ |
| **Choice** | Single-page React app (Vite), served separately or via FastAPI static files          |
| **Why**    | Minimal, fast to build, supports SSE progressive rendering easily via `EventSource`. |

### Quality Gates

| Tool               | Role                                                       |
| ------------------ | ---------------------------------------------------------- |
| `ruff`             | Linting + formatting (replaces flake8 + isort + pyupgrade) |
| `black`            | Opinionated formatter                                      |
| `mypy`             | Static type checking                                       |
| `bandit`           | Security scanning                                          |
| `pre-commit`       | Gates all of the above on commit                           |
| `pytest` + `httpx` | Unit + E2E tests                                           |
| `coverage`         | Coverage reporting (target ≥ 80%)                          |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose                               │
│                                                                     │
│  ┌──────────────┐          ┌──────────────────────────────────────┐ │
│  │   Chat UI    │  HTTP/   │         FastAPI Backend              │ │
│  │  (React/Vite)│◄─SSE────►│                                      │ │
│  │  :3000       │          │  POST /chat                          │ │
│  └──────────────┘          │  GET  /chat/stream                   │ │
│                            │  GET  /health                        │ │
│                            │  POST /ingest  (admin)               │ │
│                            └──────────────┬───────────────────────┘ │
│                                           │                         │
│                            ┌──────────────▼───────────────────────┐ │
│                            │          LangGraph Agent             │ │
│                            │                                      │ │
│                            │  ┌─────────┐                         │ │
│                            │  │ router  │ classifies intent,      │ │
│                            │  │  node   │ rejects out-of-scope    │ │
│                            │  └────┬────┘                         │ │
│                            │       │                              │ │
│                            │  ┌────▼────┐   ┌──────────────────┐  │ │
│                            │  │retrieve │──►│   ChromaDB       │  │ │
│                            │  │  node   │   │ (local vectors)  │  │ │
│                            │  └────┬────┘   └──────────────────┘  │ │
│                            │       │                              │ │
│                            │  ┌────▼──────┐                       │ │
│                            │  │  answer   │◄── Claude (Anthropic) │ │
│                            │  │   node    │    streaming          │ │
│                            │  └────┬──────┘                       │ │
│                            │       │ (conditional)                │ │
│                            │  ┌────▼──────────┐                   │ │
│                            │  │ holidays_tool │◄── Nager.Date API │ │
│                            │  │     node      │    (public)       │ │
│                            │  └────┬──────────┘                   │ │
│                            │       │                              │ │
│                            │  ┌────▼────┐                         │ │
│                            │  │fallback │  low-confidence →       │ │
│                            │  │  node   │  clarify or escalate    │ │
│                            │  └─────────┘                         │ │
│                            └──────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────┐          ┌──────────────┐   ┌─────────────────┐   │
│  │   SQLite DB  │          │  MCP Server  │   │  Observability  │   │
│  │(session mem) │          │(holidays     │   │  (structured    │   │
│  │  :aiosqlite  │          │  tool export)│   │   JSON logs)    │   │
│  └──────────────┘          └──────────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. LangGraph Agent State & Flow

```python
class AgentState(TypedDict):
    # Request
    session_id: str
    user_type: Literal["member", "staff"]
    message: str
    temperature: float
    top_p: float
    # Session continuity
    session_history: list[dict]     # [{"role": ..., "content": ...}]
    # Intent classification
    intent: str
    intent_confidence: float
    route_to_fallback: bool
    # Retrieval
    expanded_queries: list[str]
    retrieved_chunks: list[Chunk]
    retrieval_confidence: float
    # Tools
    holiday_context: str | None
    tool_calls: list[ToolCall]
    # Answer
    answer: str
    citations: list[Citation]
    confidence: float
    # Guards
    input_guard_passed: bool
    output_guard_passed: bool
    hallucination_issues: list[str]
    reflexion_attempts: int
    reflexion_exhausted: bool
    # Observability
    timing: dict[str, float]
```

### Node responsibilities

| Node             | Responsibility                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `router`         | Few-shot intent classification; follow-up resolution via session history; out_of_scope detection    |
| `query_expander` | Expand user phrasing into 3 formal retrieval queries (Haiku, prompt chaining)                       |
| `retrieve`       | 4× parallel ChromaDB searches; metadata tag filter by intent; deduplicate; route on score threshold |
| `holidays_tool`  | Nager.Date API call when intent=holiday_timing; calculates next business day; rate-limited          |
| `answer`         | Self-Ask + ReAct + CoT via structured AnswerResult; reflexion on low confidence                     |
| `fallback`       | Low retrieval confidence → warm clarifying question (Haiku, few-shot)                               |
| `rejection`      | Out-of-scope intent → polite redirect message (no LLM call)                                         |

---

## 4. Data Ingestion Pipeline

```
PDF files (data/sample_docs/)
        │
        ▼
 pdfplumber
 (extract text + page numbers)
        │
        ▼
 Semantic chunking
 (by section heading, ~500 tokens,
  50-token overlap)
        │
        ▼
 Metadata tagging
 {doc_name, page, section_heading,
  tags: [password, lockout, mfa,
         remember_me, username, ...]}
        │
        ▼
 sentence-transformers embed
 (all-MiniLM-L6-v2, local)
        │
        ▼
 ChromaDB persist (data/chroma/)
```

**PDFs ingested** (login/security scope, generated by `data/sample_docs/generate_docs.py`):

1. Login Security Items — password policy, lockout, reset flows
2. Account Lockout Guide — lockout thresholds, self-service unlock
3. MFA and Verification Codes — MFA setup, code delivery, troubleshooting
4. Remember Me and Device Trust — trusted device cadence, expiry, reset
5. Username Recovery — forgot username flows, email-based recovery
6. Phone Banking Administration — IVR user management, staff admin actions
7. Account Setup and Enrollment — first-time online access, verification steps
8. Holiday Timing and Business Days — federal holidays, reset timing expectations

---

## 5. API Contract

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
  "answer": "Absolutely — you can reset this yourself! ...",
  "citations": [
    {"doc_name": "Login — Security items", "page": 4, "section": "Account Lockout"}
  ],
  "tool_calls": [],
  "timing_ms": {
    "total": 1840,
    "retrieval": 120,
    "llm": 1680,
    "tool": 0
  }
}
```

### `GET /chat/stream?session_id=...` (SSE bonus)

```
data: {"type": "token", "content": "Absolutely"}
data: {"type": "token", "content": " — you can"}
data: {"type": "tool_start", "tool": "holidays_api"}
data: {"type": "tool_end", "result": {...}}
data: {"type": "citations", "citations": [...]}
data: {"type": "done", "timing_ms": {...}}
```

---

## 6. Project Structure

```
AITechAssessment/
├── backend/
│   ├── agent/
│   │   ├── graph.py              # LangGraph StateGraph definition + routing
│   │   ├── schemas.py            # Pydantic structured-output schemas (all nodes)
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes/
│   │   │   ├── router.py         # router + query_expander nodes
│   │   │   ├── retrieve.py       # retrieve node (tag-filtered multi-query)
│   │   │   ├── answer.py         # answer node + reflexion loop
│   │   │   ├── holidays.py       # holidays_tool node (Nager.Date API)
│   │   │   └── fallback.py       # fallback + rejection nodes
│   │   └── guards/
│   │       ├── input_guard.py    # regex pre-screen + combined Haiku LLM check
│   │       └── output_guard.py   # hallucination grounding (GroundingResult)
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan startup
│   │   ├── models.py             # Pydantic request/response models
│   │   └── routes/
│   │       ├── chat.py           # POST /chat, GET /chat/stream (SSE)
│   │       ├── health.py         # GET /health
│   │       └── ingest.py         # POST /ingest (admin)
│   ├── knowledge/
│   │   ├── ingest.py             # PDF → ChromaDB pipeline
│   │   ├── chunker.py            # Semantic chunking logic
│   │   ├── tagger.py             # Metadata tagging
│   │   └── retriever.py          # ChromaDB query wrapper
│   ├── services/
│   │   ├── conversation.py       # ConversationService (session history + graph run)
│   │   ├── sessions.py           # SQLite session/message store (aiosqlite)
│   │   └── retry.py              # tenacity backoff for LLM + API calls
│   ├── mcp_server/
│   │   └── holidays_mcp.py       # MCP server — get_federal_holidays, is_business_day, next_business_day
│   ├── scripts/
│   │   └── eval.py               # 10-prompt eval: latency, citations, SLA check
│   ├── tests/
│   │   ├── conftest.py           # shared fixtures: chroma, mocks, test_db
│   │   ├── unit/
│   │   │   ├── test_input_guard.py
│   │   │   ├── test_output_guard.py
│   │   │   ├── test_agent_graph.py
│   │   │   ├── test_retriever.py
│   │   │   ├── test_holidays_tool.py
│   │   │   ├── test_mcp_server.py
│   │   │   ├── test_chunker_tagger.py
│   │   │   ├── test_session_persistence.py
│   │   │   └── test_retry.py
│   │   └── integration/
│   │       ├── conftest.py       # integration fixtures (full wired app)
│   │       └── test_chat_api.py  # E2E: POST /chat, GET /chat/stream, /health
│   ├── data/
│   │   ├── sample_docs/          # source PDFs + generate_docs.py
│   │   └── chroma/               # ChromaDB persisted index (git-ignored)
│   ├── config.py                 # pydantic-settings (env vars + .env)
│   ├── Dockerfile
│   ├── pyproject.toml            # ruff, mypy, bandit, pytest config
│   └── .pre-commit-config.yaml
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       └── Chat.tsx
│   └── package.json
├── docker-compose.yml
├── .env.example
├── ARCHITECTURE.md               # this file
├── BACKEND_SPEC.md               # detailed TDD spec
└── README.md
```

---

## 7. Latency Budget (p95 ≤ 5s target)

| Step                       | Budget    | Notes                               |
| -------------------------- | --------- | ----------------------------------- |
| Request parsing            | ~5ms      | FastAPI                             |
| Intent classification      | ~200ms    | Claude Haiku prompt                 |
| ChromaDB retrieval         | ~50–100ms | Local, in-memory index              |
| Embedding query vector     | ~5ms      | Local model                         |
| LLM answer generation      | ~1.5–3s   | Haiku; streamed, first token ~300ms |
| Holidays API (conditional) | ~300ms    | Nager.Date, called async            |
| Response serialization     | ~5ms      |                                     |
| **Total p95 estimate**     | **~2–4s** | Comfortable headroom                |

Streaming SSE starts delivering tokens at ~300ms, so perceived latency is significantly lower.

---

## 8. Observability

- **Structured JSON logs** via `structlog`: every request logs `session_id`, `intent`, `retrieved_docs[]`, `tool_calls[]`, `timing_ms`
- **Per-node timing** captured in `AgentState.timing`
- **Retrieval hits** logged as `{doc_name, page, score}` per query
- **`scripts/eval.py`** runs all 10 sample prompts, prints latency + top-K hits, and flags any p95 breach

---

## 9. Assumptions & Trade-offs

| Assumption                                           | Rationale                                                |
| ---------------------------------------------------- | -------------------------------------------------------- |
| PDFs generated and placed in `data/sample_docs/`     | `generate_docs.py` creates 8 topic-specific PDFs covering all assessment scenarios |
| `all-MiniLM-L6-v2` is sufficient for domain accuracy | Small, curated corpus; easy to swap for a larger model   |
| SQLite over Redis for session persistence            | Simpler Docker setup; Redis is the upgrade path          |
| Anthropic API key required (not Bedrock)             | Simpler auth; documented in `.env.example`               |
| Holiday API only called for temporal queries         | Conditional routing avoids unnecessary latency           |
| Frontend is minimal (not production Blossom DS)      | Spec says "minimal chat page"; not a frontend assessment |
