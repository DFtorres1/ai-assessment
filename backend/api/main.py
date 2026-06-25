from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = structlog.get_logger()

_NO_KEY_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  WARNING: ANTHROPIC_API_KEY is not set                       ║
║  AI features are DISABLED until the key is configured.       ║
║                                                              ║
║  Fix:  export ANTHROPIC_API_KEY=sk-ant-...                   ║
║        docker compose restart backend                        ║
╚══════════════════════════════════════════════════════════════╝
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "graph"):
        import chromadb

        from adapters.session_store.sqlite import SQLiteSessionStore
        from agent.graph import create_graph
        from config import settings
        from services.conversation import ConversationService

        if not settings.anthropic_api_key:
            print(_NO_KEY_BANNER, file=sys.stderr, flush=True)
            log.warning("anthropic_api_key_missing", hint="Set ANTHROPIC_API_KEY and restart")

        chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        db = SQLiteSessionStore(db_path=settings.sqlite_db_path)
        await db.initialize()
        graph = await create_graph(chroma_client=chroma)

        conversation = ConversationService(store=db, graph=graph)

        app.state.chroma = chroma
        app.state.db = db
        app.state.graph = graph
        app.state.conversation = conversation
        _owned = True
    else:
        _owned = False

    yield

    if _owned:
        await app.state.db.close()


from api.routes import chat, health, ingest  # noqa: E402
from config import settings  # noqa: E402

app = FastAPI(
    title="Blossom Banking Helper",
    version="0.1.0",
    description="RAG-based agentic service for login & security Q&A",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingest.router)
