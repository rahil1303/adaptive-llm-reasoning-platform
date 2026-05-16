"""
Document ingestion routes.
"""

import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion import ingest_document
from app.services.vector_store import get_store

router = APIRouter()
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    # Save uploaded file
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        stats = await ingest_document(str(dest))
        return {
            "status": "ok",
            "filename": file.filename,
            "chunk_count": stats["chunk_count"],
            "total_tokens": stats["total_tokens"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def document_status():
    """Get current vector store status."""
    store = get_store()
    return store.stats()


@router.delete("/clear")
async def clear_documents():
    """Clear all ingested documents from the vector store."""
    store = get_store()
    store.clear()
    return {"status": "cleared"}
