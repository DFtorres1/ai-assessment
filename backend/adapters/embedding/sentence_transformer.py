from __future__ import annotations


class SentenceTransformerAdapter:
    """Secondary adapter: sentence-transformers embedding model behind EmbeddingPort."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._ef = SentenceTransformerEmbeddingFunction(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._ef(texts)]

    @property
    def chroma_ef(self):
        """The raw ChromaDB embedding function — only used by ChromaDBVectorStoreAdapter."""
        return self._ef
