# Blossom Banking Helper — Architecture

## System Overview

```mermaid
flowchart TD
    Client([Web / Mobile Client])
    SSE[GET /chat/stream\nSSE tokens]
    POST[POST /chat\nJSON response]

    Client -->|message| SSE
    Client -->|message| POST

    subgraph API ["FastAPI — api/"]
        IG[Input Guard\nRegex + Haiku LLM\n~150ms]
        SSE --> IG
        POST --> IG
    end

    IG -->|blocked| REJ_HTTP[HTTP 400 / SSE rejection stream]
    IG -->|passed| GRAPH

    subgraph GRAPH ["LangGraph Agent — agent/"]
        direction TB
        RN[router_node\nHaiku + Active-Prompt\n9 intents]
        QE[query_expander_node\nHaiku · 3 queries]
        RET[retrieve_node\nChromaDB cosine\ntop-5 chunks]
        HT[holidays_tool_node\nNager.Date API\ncached]
        AN[answer_node\nSonnet / Haiku\nSelf-Ask + CoT + ReAct]
        FB[fallback_node\nHaiku · clarifying Q]
        REJ[rejection_node\nstatic message]
        OG[Output Guard\nembed check + reflexion\ncitation validator]

        RN -->|in_scope| QE
        RN -->|out_of_scope| REJ
        QE --> RET
        RET -->|confident| AN
        RET -->|holiday_timing| HT
        RET -->|low_confidence| FB
        HT --> AN
        AN --> OG
    end

    OG --> RESP[Final Response\nanswer · citations · timing_ms]
    FB --> RESP
    REJ --> RESP

    subgraph STORES ["Storage"]
        CHROMA[(ChromaDB\nvector store)]
        SQLITE[(SQLite\nsession history)]
    end

    RET <-->|cosine search| CHROMA
    GRAPH <-->|get/save history| SQLITE

    subgraph MCP ["MCP Server (separate process)"]
        MCP_S[blossom-holidays\nget_federal_holidays\nis_business_day\nnext_business_day]
    end

    MCP_S -.->|same HolidaysTool| HT
```

## Component Map

| Component | File | Purpose |
|---|---|---|
| FastAPI app | `api/main.py` | Lifespan, CORS, router wiring |
| Chat routes | `api/routes/chat.py` | POST + SSE endpoints, input guard, structlog |
| Input Guard | `agent/guards/input_guard.py` | Regex + Haiku LLM scope/PII/jailbreak check |
| Output Guard | `agent/guards/output_guard.py` | Grounding + PII review + citation validation |
| Router node | `agent/nodes/router.py` | Intent classification with Active-Prompt |
| Query expander | `agent/nodes/router.py` | 3-query expansion for retrieval recall |
| Retrieve node | `agent/nodes/retrieve.py` | ChromaDB multi-query + metadata filter |
| Holidays node | `agent/nodes/holidays.py` | Nager.Date API, LRU cache, business-day logic |
| Answer node | `agent/nodes/answer.py` | Self-Ask + CoT + ReAct + Meta-Prompting |
| Fallback node | `agent/nodes/fallback.py` | Clarifying question on low retrieval confidence |
| Session store | `services/sessions.py` | SQLite async session + message history |
| Conversation service | `services/conversation.py` | Orchestrates graph + session persistence |
| MCP server | `mcp_server/holidays_mcp.py` | FastMCP holidays tools for external clients |
| Eval script | `scripts/eval.py` | 10-prompt eval with p95 latency + SLA check |

## Prompting Techniques

| Technique | Where | Effect |
|---|---|---|
| **Few-Shot** | Router | Anchors edge-case classification |
| **Active-Prompt** | Router | Selects 4 most semantically similar examples at runtime |
| **Prompt Chaining** | Query expander → retriever | Improves recall via 3 expanded queries |
| **Self-Ask** | Answer node | Decomposes question into 1–3 sub-questions |
| **Chain of Thought** | Answer node | Forces reasoning trace before answer |
| **ReAct** | Answer node | Observe (chunks) → Reason (thought) → Act (answer) |
| **Meta-Prompting** | Answer node | Haiku ≥0.90 confidence, Sonnet otherwise |
| **Directional Stimulus** | Answer node | Field descriptions guide warmth and grounding |
| **Reflexion** | Output guard | Self-critique + rewrite on low confidence/similarity |

## Data Flow (SSE streaming)

```
Client → POST /chat/stream
         ↓
    Input Guard (~150ms)
         ↓ pass
    [status] "Searching knowledge base..."
         ↓
    Graph runs (~1–4s)
         ↓
    [tool_start / tool_end]  ← if holidays API called
    [citations]
    [token] [token] [token]  ← word-by-word streaming
    [done]  timing_ms included
```
