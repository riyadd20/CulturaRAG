"""
CulturaRAG — API Routers
/query   — RAG-powered cultural Q&A
/ingest  — Document ingestion (text, PDF, DOCX, file upload)
/feedback — RLHF human feedback collection
/status  — Index health & stats
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.responses import JSONResponse
from typing import Optional
from loguru import logger

from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestTextRequest, IngestResponse,
    FeedbackRequest, FeedbackResponse,
    IndexStatusResponse,
)
from app.services.rag_chain import get_rag_chain
from app.services.ingestion import get_ingestion_service
from app.services.feedback import get_feedback_service
from app.services.vector_store import get_vector_store


# ── Query Router ──────────────────────────────────────────────────────────────
query_router = APIRouter(prefix="/query", tags=["Cultural Q&A"])


@query_router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a cultural question",
    description=(
        "Submit a question about any world culture, tradition, or language. "
        "The system retrieves the most relevant documents from the FAISS index "
        "and generates a grounded answer using Google Gemini."
    ),
)
async def ask_question(request: QueryRequest):
    try:
        chain = get_rag_chain()
        return chain.query(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── Ingestion Router ──────────────────────────────────────────────────────────
ingest_router = APIRouter(prefix="/ingest", tags=["Document Ingestion"])


@ingest_router.post(
    "/text",
    response_model=IngestResponse,
    summary="Ingest raw text",
    description="Add raw cultural text directly to the FAISS knowledge base.",
)
async def ingest_text(request: IngestTextRequest):
    try:
        svc = get_ingestion_service()
        result = svc.ingest_text(
            text=request.text,
            source=request.source,
            culture=request.culture,
            language=request.language,
            tags=request.tags,
        )
        return IngestResponse(
            success=True,
            chunks_added=result["chunks_added"],
            source=result["source"],
            message=f"Successfully ingested {result['chunks_added']} chunks from '{result['source']}'",
        )
    except Exception as e:
        logger.exception(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest a PDF or DOCX file",
    description="Upload a PDF or DOCX cultural document to ingest into the knowledge base.",
)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(...),
    culture: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default="en"),
):
    svc = get_ingestion_service()
    content = await file.read()
    fname = file.filename or ""

    try:
        if fname.lower().endswith(".pdf"):
            result = svc.ingest_pdf(content, source, culture, language)
        elif fname.lower().endswith(".docx"):
            result = svc.ingest_docx(content, source, culture, language)
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF and DOCX files are supported.",
            )
        return IngestResponse(
            success=True,
            chunks_added=result["chunks_added"],
            source=result["source"],
            message=f"File '{fname}' ingested: {result['chunks_added']} chunks added.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.post(
    "/seed",
    summary="Seed built-in cultural data",
    description="Populate the index with built-in sample cultural documents (useful for demos).",
)
async def seed_data():
    svc = get_ingestion_service()
    total = svc.ingest_sample_cultural_data()
    return {"status": "ok", "chunks_added": total, "message": "Sample cultural data seeded."}


# ── Feedback Router ───────────────────────────────────────────────────────────
feedback_router = APIRouter(prefix="/feedback", tags=["RLHF Feedback"])


@feedback_router.post(
    "/",
    response_model=FeedbackResponse,
    summary="Submit feedback on a response",
    description=(
        "Rate a RAG response and optionally provide a corrected answer. "
        "Feedback is stored in a JSONL file for RLHF / LoRA fine-tuning pipelines."
    ),
)
async def submit_feedback(request: FeedbackRequest):
    try:
        svc = get_feedback_service()
        return svc.record(request)
    except Exception as e:
        logger.exception(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.get(
    "/stats",
    summary="Feedback statistics",
    description="Retrieve aggregate feedback statistics (ratings, thumbs, correction pairs).",
)
async def feedback_stats():
    svc = get_feedback_service()
    return svc.get_summary_stats()


@feedback_router.get(
    "/export",
    summary="Export preference pairs for LoRA training",
    description="Export feedback entries with corrected answers as DPO/SFT preference pairs.",
)
async def export_pairs():
    svc = get_feedback_service()
    pairs = svc.export_preference_pairs()
    return {"total_pairs": len(pairs), "pairs": pairs}


# ── Status Router ─────────────────────────────────────────────────────────────
status_router = APIRouter(prefix="/status", tags=["Index Status"])


@status_router.get(
    "/",
    response_model=IndexStatusResponse,
    summary="Vector index health & statistics",
)
async def get_status():
    vs = get_vector_store()
    stats = vs.get_stats()
    return IndexStatusResponse(**stats)
