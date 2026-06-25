import time
from pathlib import Path

from fastapi import APIRouter, Request

from api.models import IngestResponse
from config import settings

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    t_start = time.monotonic()
    pdf_dir = Path(settings.pdf_dir)
    chroma_client = request.app.state.chroma

    from knowledge.ingest import ingest_pdfs

    result = await ingest_pdfs(pdf_dir=pdf_dir, chroma_client=chroma_client)

    return IngestResponse(
        chunks_indexed=result.get("indexed", 0),
        documents_processed=[],
        duration_ms=round((time.monotonic() - t_start) * 1000),
    )
