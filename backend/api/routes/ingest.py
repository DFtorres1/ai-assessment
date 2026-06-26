import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from api.models import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, file: Annotated[UploadFile, File(...)]) -> IngestResponse:
    t_start = time.monotonic()
    chroma_client = request.app.state.chroma

    from knowledge.ingest import ingest_pdfs

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dest = tmp_path / (file.filename or "upload.pdf")
        dest.write_bytes(await file.read())

        result = await ingest_pdfs(pdf_dir=tmp_path, chroma_client=chroma_client)

    return IngestResponse(
        chunks_indexed=result.get("indexed", 0),
        documents_processed=result.get("documents", []),
        duration_ms=round((time.monotonic() - t_start) * 1000),
    )
