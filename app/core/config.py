"""
CulturaRAG — Core Configuration
Loads all settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List
import os


class Settings(BaseSettings):
    # ── Gemini ──────────────────────────────────────────────
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")

    # ── Embeddings ──────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL",
    )

    # ── FAISS ───────────────────────────────────────────────
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    faiss_top_k: int = Field(default=5, env="FAISS_TOP_K")

    # ── RAG ─────────────────────────────────────────────────
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    max_context_tokens: int = Field(default=4096, env="MAX_CONTEXT_TOKENS")

    # ── API ─────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS",
    )

    # ── RLHF Feedback ───────────────────────────────────────
    feedback_log_path: str = Field(
        default="./data/feedback_log.jsonl", env="FEEDBACK_LOG_PATH"
    )
    enable_feedback_collection: bool = Field(
        default=True, env="ENABLE_FEEDBACK_COLLECTION"
    )

    # ── LoRA Fine-tuning ────────────────────────────────────
    base_model_for_lora: str = Field(
        default="google/gemma-2b", env="BASE_MODEL_FOR_LORA"
    )
    lora_output_dir: str = Field(
        default="./data/lora_adapters", env="LORA_OUTPUT_DIR"
    )
    lora_rank: int = Field(default=16, env="LORA_RANK")
    lora_alpha: int = Field(default=32, env="LORA_ALPHA")
    lora_dropout: float = Field(default=0.05, env="LORA_DROPOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
