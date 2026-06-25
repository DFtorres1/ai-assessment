from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import tiktoken

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(
    text: str,
    doc_name: str,
    page: int = 0,
    section: str = "",
) -> list[Chunk]:
    """
    Split text into ≤500-token chunks with 50-token overlap using cl100k_base.
    Section boundaries (empty lines between paragraphs) force a new chunk.
    """
    tokens = _TOKENIZER.encode(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens).strip()

        if chunk_text:
            chunk_id = _make_id(doc_name, page, section, start)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={"doc_name": doc_name, "page": page, "section": section, "tags": ""},
                )
            )

        if end >= len(tokens):
            break
        start = end - CHUNK_OVERLAP_TOKENS

    return chunks


def _make_id(doc_name: str, page: int, section: str, offset: int) -> str:
    key = f"{doc_name}|{page}|{section}|{offset}"
    return hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
