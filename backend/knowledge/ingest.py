from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
import structlog

from knowledge.chunker import Chunk, chunk_document
from knowledge.tagger import tag_chunks

log = structlog.get_logger()

# Heuristic: a line is a section header if it's ≤80 chars, ends without punctuation,
# and is not purely numeric.
_SECTION_RE = re.compile(r"^[A-Z][^.!?]{2,79}$")


def _detect_section(line: str) -> str | None:
    line = line.strip()
    if _SECTION_RE.match(line) and not line.replace(" ", "").isdigit():
        return line
    return None


def _extract_sections(page_text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, text) pairs from a page's text."""
    sections: list[tuple[str, str]] = []
    current_section = ""
    current_lines: list[str] = []

    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        detected = _detect_section(line)
        if detected:
            if current_lines:
                sections.append((current_section, " ".join(current_lines)))
            current_section = detected
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, " ".join(current_lines)))

    return sections if sections else [("", page_text.replace("\n", " ").strip())]


async def ingest_pdfs(pdf_dir: Path, chroma_client: Any) -> dict[str, Any]:
    """
    PDF → ChromaDB ingestion pipeline.
    Reads all PDFs from pdf_dir, chunks, tags, embeds, and stores.
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        log.warning("ingest.no_pdfs", directory=str(pdf_dir))
        return {"status": "no_pdfs", "indexed": 0}

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_or_create_collection(
        name="blossom_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        doc_name = pdf_path.stem
        log.info("ingest.processing", pdf=doc_name)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                for section, section_text in _extract_sections(page_text):
                    chunks = chunk_document(
                        section_text, doc_name=doc_name, page=page_num, section=section
                    )
                    all_chunks.extend(chunks)

    tagged = tag_chunks(all_chunks)

    if not tagged:
        return {"status": "no_content", "indexed": 0}

    batch_size = 100
    total = 0
    for i in range(0, len(tagged), batch_size):
        batch = tagged[i : i + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        total += len(batch)
        log.info("ingest.batch_added", count=len(batch), total=total)

    return {"status": "ok", "indexed": total}
