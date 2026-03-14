"""
CulturaRAG — Pydantic Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
import uuid


# ── Query Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Cultural question to ask the RAG system",
        example="What are the main traditions of the Japanese Obon festival?",
    )
    language: Optional[str] = Field(
        default="en",
        description="ISO 639-1 language code for the response",
        example="en",
    )
    culture_filter: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific culture",
        example="Japanese",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )


class SourceDocument(BaseModel):
    content: str
    source: str
    culture: Optional[str] = None
    language: Optional[str] = None
    score: float = Field(description="Cosine similarity score")


class QueryResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    sources: List[SourceDocument]
    language: str
    culture_filter: Optional[str]
    model_used: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Ingestion Models ─────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw cultural text to ingest")
    source: str = Field(..., description="Name or URL of the source")
    culture: Optional[str] = Field(default=None, description="Associated culture/region")
    language: Optional[str] = Field(default="en", description="Language of the text")
    tags: Optional[List[str]] = Field(default=[], description="Optional metadata tags")


class IngestResponse(BaseModel):
    success: bool
    chunks_added: int
    source: str
    message: str


# ── RLHF Feedback Models ─────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="ID of the query to give feedback on")
    rating: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="Rating from 1 (poor) to 5 (excellent)"
    )
    thumbs: Optional[Literal["up", "down"]] = Field(
        default=None, description="Quick thumbs up/down feedback"
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional free-text feedback for RLHF training",
    )
    corrected_answer: Optional[str] = Field(
        default=None,
        description="If the answer was wrong, provide the correct answer here",
    )


class FeedbackResponse(BaseModel):
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    status: str = "recorded"
    message: str = "Thank you! Your feedback will improve future responses."
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Index Status ─────────────────────────────────────────────────────────────

class IndexStatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    cultures_indexed: List[str]
    languages_indexed: List[str]
    index_size_mb: float
    last_updated: Optional[datetime]
