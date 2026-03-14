"""
CulturaRAG — RAG Chain Service
Orchestrates retrieval → prompt construction → Groq/LLaMA generation.
"""

import time
import os
from typing import List, Optional, Tuple, Dict, Any

from loguru import logger

from app.core.config import get_settings
from app.services.vector_store import get_vector_store
from app.models.schemas import QueryRequest, QueryResponse, SourceDocument

settings = get_settings()

SYSTEM_PROMPT = """You are CulturaRAG, an expert multilingual cultural knowledge assistant.
Your role is to provide accurate, nuanced, and respectful explanations of world cultures,
traditions, languages, and practices.

Guidelines:
- Ground your answer ONLY in the provided context documents
- If the context is insufficient, say so clearly rather than hallucinating
- Respect cultural sensitivity and avoid stereotyping
- When relevant, highlight regional variations within a culture
- If asked in a specific language, respond in that language
- Cite which source documents informed your answer"""

QUERY_TEMPLATE = """CONTEXT DOCUMENTS:
{context}

---
USER QUESTION: {question}
RESPONSE LANGUAGE: {language}
{culture_hint}

Please provide a thorough, culturally sensitive answer grounded in the context above."""


def _build_context(retrieved: List[Tuple[Dict, float]]) -> str:
    lines = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        lines.append(
            f"[Document {i}] (Source: {chunk.get('source', 'unknown')}, "
            f"Culture: {chunk.get('culture', 'N/A')}, "
            f"Relevance: {score:.3f})\n{chunk['content']}"
        )
    return "\n\n".join(lines)


def _language_name(code: str) -> str:
    mapping = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "zh": "Chinese (Mandarin)", "ja": "Japanese",
        "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese",
        "ko": "Korean", "ru": "Russian", "sw": "Swahili",
    }
    return mapping.get(code.lower(), code)


class RAGChain:
    def __init__(self):
        self._llm = None
        self.vector_store = get_vector_store()

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.environ.get("GROQ_API_KEY"),
                temperature=0.3,
            )
            logger.info("Groq LLM initialized: llama-3.1-8b-instant")
        return self._llm

    def query(self, request: QueryRequest) -> QueryResponse:
        t_start = time.perf_counter()

        retrieved = self.vector_store.search(
            query=request.question,
            top_k=request.top_k or settings.faiss_top_k,
            culture_filter=request.culture_filter,
        )
        logger.info(f"Retrieved {len(retrieved)} chunks for: '{request.question[:60]}…'")

        context_str = _build_context(retrieved) if retrieved else (
            "No relevant cultural documents found in the knowledge base."
        )
        culture_hint = (
            f"CULTURE FOCUS: {request.culture_filter}" if request.culture_filter else ""
        )
        full_prompt = QUERY_TEMPLATE.format(
            context=context_str,
            question=request.question,
            language=_language_name(request.language or "en"),
            culture_hint=culture_hint,
        )

        llm = self._get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=full_prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content

        latency_ms = (time.perf_counter() - t_start) * 1000

        sources = [
            SourceDocument(
                content=chunk["content"][:400] + ("…" if len(chunk["content"]) > 400 else ""),
                source=chunk.get("source", "unknown"),
                culture=chunk.get("culture"),
                language=chunk.get("language"),
                score=score,
            )
            for chunk, score in retrieved
        ]

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            language=request.language or "en",
            culture_filter=request.culture_filter,
            model_used="llama-3.1-8b-instant",
            latency_ms=round(latency_ms, 2),
        )


_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
