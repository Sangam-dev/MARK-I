"""Upload API and document management for the RAG subsystem.

This router is the HTTP face of the **upload pipeline**, which is
independent of the conversation pipeline: it drives
:class:`~memory.rag.ingest.IngestService` and never emits a bus event or
touches the Conversation Manager. The two pipelines meet only at the
shared vector database.

Endpoints::

    POST   /api/rag/upload          multipart file -> indexed document
    GET    /api/rag/documents       list indexed documents
    DELETE /api/rag/documents/{id}  remove a document and its chunks
    GET    /api/rag/search          debug/inspection retrieval
    GET    /api/rag/stats           index + config snapshot

``multipart/form-data`` needs ``python-multipart`` installed; without it
FastAPI raises at import of the route, so the dependency is declared in
pyproject.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from api.schemas import (
    RAGDeleteResult,
    RAGDocumentOut,
    RAGSearchHit,
    RAGSearchResponse,
    RAGStats,
    RAGUploadResult,
)

router = APIRouter(prefix="/rag", tags=["rag"])

logger = logging.getLogger("kancha.api.routes.rag")


def _require_rag(request: Request):
    """Return the live RAGManager or raise a clean 503.

    RAG is optional — the pipeline boots without it when disabled or when
    its startup failed. Every endpoint here therefore has to answer the
    "what if it isn't there" question explicitly.
    """
    pipeline = request.app.state.pipeline
    manager = getattr(pipeline, "rag", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG is not available. It is either disabled "
                "(KANCHA_RAG_ENABLED=0) or failed to initialise — check the "
                "backend logs."
            ),
        )
    return manager


def _require_ingest(request: Request):
    """Return the live IngestService or raise a clean 503."""
    pipeline = request.app.state.pipeline
    service = getattr(pipeline, "rag_ingest", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Document upload is unavailable because RAG is not running.",
        )
    return service


# ── Upload pipeline ──────────────────────────────────────────────────


@router.post("/upload", response_model=RAGUploadResult)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, TXT, Markdown or DOCX file"),
    title: str | None = Form(default=None),
) -> RAGUploadResult:
    """Upload one file and index it into the vector database.

    The full chain — temporary storage, loading, text extraction,
    chunking, embedding, indexing — runs inside
    :meth:`IngestService.ingest_upload`. This handler only translates the
    resulting report into an HTTP response.
    """
    service = _require_ingest(request)

    try:
        data = await file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")
    finally:
        await file.close()

    report = await service.ingest_upload(
        filename=file.filename or "upload",
        data=data,
        title=(title or "").strip() or None,
    )

    if not report.success:
        # 415 for a format we do not handle, 400 for everything else the
        # caller could fix (empty file, oversized, unreadable).
        status = 415 if "Unsupported file type" in report.message else 400
        raise HTTPException(status_code=status, detail=report.message)

    logger.info(
        "Upload indexed: %s -> %s (%d chunks)",
        report.filename,
        report.document_id,
        report.chunks_indexed,
    )
    return RAGUploadResult(**report.to_dict())


# ── Document management ──────────────────────────────────────────────


@router.get("/documents", response_model=list[RAGDocumentOut])
async def list_documents(request: Request) -> list[RAGDocumentOut]:
    """List every indexed document, newest first."""
    manager = _require_rag(request)
    documents = await manager.list_documents()
    return [RAGDocumentOut(**document) for document in documents]


@router.delete("/documents/{document_id}", response_model=RAGDeleteResult)
async def delete_document(request: Request, document_id: str) -> RAGDeleteResult:
    """Delete a document and every chunk derived from it."""
    manager = _require_rag(request)

    existing = await manager.get_document(document_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id!r}")

    removed = await manager.delete_document(document_id)
    return RAGDeleteResult(
        document_id=document_id,
        title=str(existing.get("title", "")),
        chunks_removed=removed,
    )


# ── Inspection ───────────────────────────────────────────────────────


@router.get("/search", response_model=RAGSearchResponse)
async def search(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int | None = Query(default=None, ge=1, le=50),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> RAGSearchResponse:
    """Run a retrieval directly against the index.

    Exists for debugging and for a future UI panel — the conversation
    path does **not** call this; it goes through the Memory Router and
    the RAG Manager in-process.
    """
    manager = _require_rag(request)
    hits = await manager.retrieve(q, top_k=top_k, threshold=threshold)
    return RAGSearchResponse(
        query=q,
        count=len(hits),
        results=[RAGSearchHit(**hit.to_dict()) for hit in hits],
    )


@router.get("/stats", response_model=RAGStats)
async def stats(request: Request) -> RAGStats:
    """Index size and effective configuration.

    Unlike the other endpoints this one answers even when RAG is off, so
    the UI can show *why* upload is unavailable instead of a bare error.
    """
    pipeline = request.app.state.pipeline
    manager = getattr(pipeline, "rag", None)
    ingest = getattr(pipeline, "rag_ingest", None)

    if manager is None:
        return RAGStats(ready=False, enabled=False)

    snapshot = await manager.stats()
    if ingest is not None:
        snapshot["supported_extensions"] = list(ingest.supported_extensions())
        snapshot["max_upload_bytes"] = ingest.max_upload_bytes
    return RAGStats(**snapshot)
