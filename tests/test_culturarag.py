"""
CulturaRAG — Test Suite
Run: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """FastAPI test client with mocked external services."""
    with patch("app.services.vector_store.VectorStoreService._load_or_create_index"):
        with patch("app.services.vector_store.VectorStoreService._persist"):
            from app.main import app
            with TestClient(app) as c:
                yield c


@pytest.fixture
def sample_query():
    return {
        "question": "What are the main traditions of Diwali?",
        "language": "en",
        "culture_filter": "Indian",
        "top_k": 3,
    }


@pytest.fixture
def sample_ingest():
    return {
        "text": (
            "The Maori haka is a ceremonial dance from New Zealand, traditionally "
            "performed by the indigenous Maori people. It involves vigorous body "
            "movements, stomping, and chanting, originally used to welcome guests, "
            "celebrate achievements, or prepare for battle."
        ),
        "source": "Maori Cultural Guide",
        "culture": "Maori",
        "language": "en",
        "tags": ["dance", "ceremony", "New Zealand"],
    }


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_valid(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="Tell me about Diwali", language="en")
        assert req.question == "Tell me about Diwali"
        assert req.language == "en"

    def test_query_request_too_short(self):
        from pydantic import ValidationError
        from app.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")  # min_length=3 but also too vague

    def test_feedback_rating_range(self):
        from pydantic import ValidationError
        from app.models.schemas import FeedbackRequest
        with pytest.raises(ValidationError):
            FeedbackRequest(query_id="abc", rating=6)  # max is 5

    def test_feedback_valid(self):
        from app.models.schemas import FeedbackRequest
        req = FeedbackRequest(
            query_id="test-id-123",
            rating=4,
            thumbs="up",
            comment="Very accurate and helpful!",
        )
        assert req.rating == 4
        assert req.thumbs == "up"


# ── Ingestion Tests ───────────────────────────────────────────────────────────

class TestIngestion:
    def test_split_text_short(self):
        from app.services.ingestion import _split_text
        short = "Hello world. This is a short text."
        chunks = _split_text(short, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == short

    def test_split_text_long(self):
        from app.services.ingestion import _split_text
        long_text = "This is a sentence. " * 100  # ~2000 chars
        chunks = _split_text(long_text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk should be under chunk_size + some overlap buffer
        for c in chunks:
            assert len(c) <= 400  # with some tolerance

    def test_ingest_text_creates_chunks(self, sample_ingest):
        mock_vs = MagicMock()
        mock_vs.add_chunks.return_value = 1

        with patch("app.services.ingestion.get_vector_store", return_value=mock_vs):
            from app.services.ingestion import IngestionService
            svc = IngestionService()
            result = svc.ingest_text(
                text=sample_ingest["text"],
                source=sample_ingest["source"],
                culture=sample_ingest["culture"],
                language=sample_ingest["language"],
            )
            assert result["source"] == "Maori Cultural Guide"
            assert mock_vs.add_chunks.called


# ── Feedback Tests ────────────────────────────────────────────────────────────

class TestFeedback:
    def test_record_feedback(self, tmp_path):
        from app.services.feedback import FeedbackService
        from app.models.schemas import FeedbackRequest

        svc = FeedbackService.__new__(FeedbackService)
        svc.log_path = tmp_path / "feedback.jsonl"
        svc.enabled = True

        req = FeedbackRequest(
            query_id="q-test-001",
            rating=5,
            thumbs="up",
            comment="Excellent cultural explanation!",
        )
        resp = svc.record(req)
        assert resp.query_id == "q-test-001"
        assert resp.status == "recorded"
        assert svc.log_path.exists()

    def test_export_preference_pairs(self, tmp_path):
        from app.services.feedback import FeedbackService

        svc = FeedbackService.__new__(FeedbackService)
        svc.log_path = tmp_path / "feedback.jsonl"
        svc.enabled = True

        # Write a fake entry with corrected_answer
        entry = {
            "feedback_id": "fb-001",
            "query_id": "q-001",
            "rating": 2,
            "corrected_answer": "The correct answer about Diwali is …",
            "timestamp": "2024-01-01T00:00:00",
        }
        with open(svc.log_path, "w") as f:
            f.write(json.dumps(entry) + "\n")

        pairs = svc.export_preference_pairs()
        assert len(pairs) == 1
        assert pairs[0]["corrected_answer"].startswith("The correct answer")

    def test_summary_stats_empty(self, tmp_path):
        from app.services.feedback import FeedbackService

        svc = FeedbackService.__new__(FeedbackService)
        svc.log_path = tmp_path / "nonexistent.jsonl"
        svc.enabled = True

        stats = svc.get_summary_stats()
        assert stats["total"] == 0
        assert stats["avg_rating"] is None


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_settings(self):
        from app.core.config import Settings
        s = Settings()
        assert s.faiss_top_k == 5
        assert s.chunk_size == 800
        assert s.lora_rank == 16
        assert s.gemini_model == "gemini-1.5-pro"


# ── API Endpoint Smoke Tests ──────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "CulturaRAG" in resp.text

    def test_status_endpoint(self, client):
        with patch("app.api.routes.get_vector_store") as mock_vs:
            mock_vs.return_value.get_stats.return_value = {
                "total_chunks": 42,
                "total_documents": 6,
                "cultures_indexed": ["Indian", "Japanese"],
                "languages_indexed": ["en"],
                "index_size_mb": 1.2,
                "last_updated": None,
            }
            resp = client.get("/api/v1/status/")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_chunks"] == 42

    def test_ingest_text_endpoint(self, client, sample_ingest):
        with patch("app.api.routes.get_ingestion_service") as mock_svc:
            mock_svc.return_value.ingest_text.return_value = {
                "chunks_added": 2,
                "source": "Maori Cultural Guide",
            }
            resp = client.post("/api/v1/ingest/text", json=sample_ingest)
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["chunks_added"] == 2

    def test_feedback_endpoint(self, client):
        with patch("app.api.routes.get_feedback_service") as mock_svc:
            from app.models.schemas import FeedbackResponse
            mock_svc.return_value.record.return_value = FeedbackResponse(
                feedback_id="fb-test",
                query_id="q-test",
            )
            resp = client.post("/api/v1/feedback/", json={
                "query_id": "q-test",
                "rating": 4,
                "thumbs": "up",
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "recorded"
