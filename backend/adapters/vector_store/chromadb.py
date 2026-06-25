from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.state import Chunk


@dataclass
class _RawChunk:
    id: str
    text: str
    doc_name: str
    page: int
    section: str
    tags: str
    score: float


class ChromaDBVectorStoreAdapter:
    """Secondary adapter: ChromaDB vector store behind VectorStorePort."""

    def __init__(
        self,
        chroma_client: Any,
        embed_fn: Any,
        collection_name: str = "blossom_knowledge",
    ) -> None:
        self._client = chroma_client
        self._embed_fn = embed_fn
        self._collection_name = collection_name

    def _collection(self) -> Any:
        return self._client.get_or_create_collection(
            self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _parse(self, results: dict[str, Any]) -> list[_RawChunk]:
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            _RawChunk(
                id=id_,
                text=doc,
                doc_name=meta.get("doc_name", ""),
                page=int(meta.get("page", 0)),
                section=meta.get("section", ""),
                tags=meta.get("tags", ""),
                score=round(max(0.0, 1.0 - dist), 4),
            )
            for id_, doc, meta, dist in zip(ids, docs, metas, distances, strict=False)
        ]

    def _tag_match(self, chunk: _RawChunk, tags: list[str]) -> bool:
        return any(t in chunk.tags for t in tags)

    def _to_domain(self, rc: _RawChunk) -> Chunk:
        return Chunk(
            doc_name=rc.doc_name,
            page=rc.page,
            section=rc.section,
            text=rc.text,
            score=rc.score,
            tags=[t for t in rc.tags.split(",") if t] if rc.tags else [],
        )

    async def search(
        self,
        query: str,
        k: int = 5,
        tags: list[str] | None = None,
    ) -> list[Chunk]:
        col = self._collection()
        total = col.count()
        if total == 0:
            return []
        n = min(total, k * 3 if tags else k)
        results = col.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = self._parse(results)
        if tags:
            chunks = [c for c in chunks if self._tag_match(c, tags)]
        return [self._to_domain(c) for c in chunks[:k]]

    async def search_multi(
        self,
        queries: list[str],
        k: int = 5,
        tags: list[str] | None = None,
    ) -> list[Chunk]:
        best: dict[str, _RawChunk] = {}
        for query in queries:
            col = self._collection()
            total = col.count()
            if total == 0:
                continue
            n = min(total, k * 3 if tags else k)
            results = col.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            raw_chunks = self._parse(results)
            if tags:
                raw_chunks = [c for c in raw_chunks if self._tag_match(c, tags)]
            for chunk in raw_chunks[:k]:
                if chunk.id not in best or chunk.score > best[chunk.id].score:
                    best[chunk.id] = chunk
        sorted_raw = sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]
        return [self._to_domain(c) for c in sorted_raw]
