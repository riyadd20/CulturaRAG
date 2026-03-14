"""
CulturaRAG — RLHF Feedback Service
Collects human feedback on RAG responses for future RLHF / fine-tuning pipelines.

The feedback log (JSONL) can be consumed by:
  - Direct RLHF training via reward model training
  - LoRA fine-tuning dataset construction (corrected_answer → preference pairs)
  - Analytics dashboards to track answer quality over time
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from loguru import logger
from app.core.config import get_settings
from app.models.schemas import FeedbackRequest, FeedbackResponse

settings = get_settings()


class FeedbackService:
    """
    Persists human feedback to a JSONL file for RLHF pipeline consumption.

    JSONL Schema (one JSON object per line):
    {
        "feedback_id": str,
        "query_id": str,
        "rating": int (1-5),
        "thumbs": "up" | "down" | null,
        "comment": str | null,
        "corrected_answer": str | null,   # <-- used for preference pairs in LoRA fine-tuning
        "timestamp": ISO8601 str
    }
    """

    def __init__(self):
        self.log_path = Path(settings.feedback_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = settings.enable_feedback_collection

    def record(self, request: FeedbackRequest) -> FeedbackResponse:
        """Append a feedback record to the JSONL log."""
        if not self.enabled:
            logger.warning("Feedback collection is disabled.")
            return FeedbackResponse(
                query_id=request.query_id,
                status="disabled",
                message="Feedback collection is currently disabled.",
            )

        feedback_id = str(uuid.uuid4())
        record: Dict[str, Any] = {
            "feedback_id": feedback_id,
            "query_id": request.query_id,
            "rating": request.rating,
            "thumbs": request.thumbs,
            "comment": request.comment,
            "corrected_answer": request.corrected_answer,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Feedback recorded: {feedback_id} (rating={request.rating})")
        except Exception as e:
            logger.error(f"Failed to write feedback: {e}")
            raise

        return FeedbackResponse(feedback_id=feedback_id, query_id=request.query_id)

    def export_preference_pairs(self) -> List[Dict[str, Any]]:
        """
        Export feedback entries that have corrected_answer set,
        formatted as preference pairs for LoRA / DPO fine-tuning.

        Returns a list of dicts:
        {
            "prompt": str,        # original question (query_id placeholder)
            "chosen": str,        # corrected_answer (human preferred)
            "rejected": str,      # original answer placeholder
        }
        """
        if not self.log_path.exists():
            return []

        pairs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("corrected_answer"):
                        pairs.append({
                            "query_id": entry["query_id"],
                            "corrected_answer": entry["corrected_answer"],
                            "rating": entry["rating"],
                            "timestamp": entry["timestamp"],
                        })
                except json.JSONDecodeError:
                    continue
        return pairs

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about collected feedback."""
        if not self.log_path.exists():
            return {"total": 0, "avg_rating": None, "thumbs_up": 0, "thumbs_down": 0}

        ratings, thumbs_up, thumbs_down, corrections = [], 0, 0, 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ratings.append(entry.get("rating", 0))
                    if entry.get("thumbs") == "up":
                        thumbs_up += 1
                    elif entry.get("thumbs") == "down":
                        thumbs_down += 1
                    if entry.get("corrected_answer"):
                        corrections += 1
                except json.JSONDecodeError:
                    continue

        return {
            "total": len(ratings),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "correction_pairs": corrections,
        }


_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
