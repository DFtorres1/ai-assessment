from __future__ import annotations

# Backward-compatibility shim — implementation moved to adapters/vector_store/chromadb.py
from adapters.vector_store.chromadb import ChromaDBVectorStoreAdapter as Retriever
from adapters.vector_store.chromadb import _RawChunk as RetrievedChunk

__all__ = ["Retriever", "RetrievedChunk"]
