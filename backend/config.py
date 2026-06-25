from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Point HuggingFace and SentenceTransformers at the local cache so the
# all-MiniLM-L6-v2 model loads without hitting the network or /root/.cache.
_LOCAL_HF_CACHE = str(Path(__file__).parent / ".cache" / "huggingface")
os.environ.setdefault("HF_HOME", _LOCAL_HF_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_HF_CACHE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str = Field(default="")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Models
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6-20250514"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "blossom_knowledge"
    pdf_dir: str = "./data/sample_docs"

    # SQLite
    sqlite_db_path: str = "./data/sessions.db"

    # Retrieval thresholds
    # 0.50 is appropriate for all-MiniLM-L6-v2 cosine similarity; recalibrate if
    # switching embedding models or significantly expanding the corpus
    retrieval_routing_threshold: float = 0.50
    reflexion_confidence_threshold: float = 0.60
    reflexion_embedding_threshold: float = 0.60
    max_reflexion_attempts: int = 2

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Session
    session_history_max_turns: int = 3  # = 6 messages (user + assistant per turn)

    # CORS — comma-separated list of allowed origins for local dev
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:5173"


settings = Settings()
