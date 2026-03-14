# 🌍 CulturaRAG — AI Knowledge Explorer for World Cultures & Languages

A production-ready **Retrieval-Augmented Generation (RAG)** system that combines
**FAISS vector search**, **LangChain orchestration**, and **Google Gemini** to deliver
nuanced, multilingual cultural insights — with a built-in **RLHF feedback loop** and
**LoRA fine-tuning** pipeline.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                       │
│                                                           │
│  POST /query  ──►  RAGChain                               │
│                      │                                    │
│                      ├─► FAISS Vector Search              │
│                      │     (multilingual SentenceTransformer)
│                      │                                    │
│                      └─► Google Gemini (LangChain)        │
│                            context-grounded generation    │
│                                                           │
│  POST /ingest ──►  IngestionService                       │
│                      text chunker → FAISS index           │
│                                                           │
│  POST /feedback ─► FeedbackService → JSONL log            │
│                      │                                    │
│                      └─► LoRA fine-tuning pipeline        │
│                           (PEFT, preference pairs)        │
└──────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM Orchestration | LangChain |
| Language Model | Google Gemini 1.5 Pro |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers) |
| RLHF / Fine-tuning | PEFT / LoRA (`peft`, `transformers`, `datasets`) |
| Document Parsing | PyPDF, python-docx |
| Config | Pydantic Settings + dotenv |
| Logging | Loguru |
| Testing | Pytest |
| Containerization | Docker + docker-compose |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/culturarag.git
cd culturarag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Run the Server

```bash
python run.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

On first launch, the server auto-seeds the index with 6 diverse cultural documents
(Diwali, Hanami, Día de los Muertos, Ramadan, Ubuntu philosophy, Chinese New Year).

---

## API Endpoints

### Ask a Cultural Question
```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the significance of the sakura in Japanese culture?",
    "language": "en",
    "culture_filter": "Japanese",
    "top_k": 5
  }'
```

### Multilingual Query (respond in Spanish)
```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the Ubuntu philosophy from Africa",
    "language": "es"
  }'
```

### Ingest New Cultural Text
```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Scottish Highland Games feature athletic events like caber tossing...",
    "source": "Scottish Culture Guide",
    "culture": "Scottish",
    "language": "en"
  }'
```

### Upload a PDF
```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@my_cultural_doc.pdf" \
  -F "source=My PDF Source" \
  -F "culture=Greek"
```

### Submit RLHF Feedback
```bash
curl -X POST http://localhost:8000/api/v1/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "returned-query-id-here",
    "rating": 4,
    "thumbs": "up",
    "comment": "Very thorough explanation of the festival origins",
    "corrected_answer": null
  }'
```

### Check Index Status
```bash
curl http://localhost:8000/api/v1/status/
```

---

## RLHF & LoRA Fine-tuning Pipeline

```
User rates a response
       │
       ▼
FeedbackService.record()
  → appends to data/feedback_log.jsonl
  → {query_id, rating, corrected_answer, ...}
       │
       ▼
GET /api/v1/feedback/export
  → returns preference pairs (chosen / rejected)
       │
       ▼
python -m app.utils.lora_finetune \
  --dataset ./data/feedback_log.jsonl \
  --base-model google/gemma-2b \
  --output-dir ./data/lora_adapters \
  --rank 16 --alpha 32 --epochs 3
       │
       ▼
LoRA adapter saved → load adapter at inference time
```

### When to Fine-tune
- Collect at least **100+ corrected answers** (feedback with `corrected_answer` set)
- High-confidence pairs: `rating >= 4` are selected by default
- Run as an offline batch job (separate from the API server)

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## Docker Deployment

```bash
# Set your API key
export GEMINI_API_KEY=your_key_here

# Build and run
docker-compose up --build

# The FAISS index & feedback logs persist in ./data/
```

---

## Extending the Knowledge Base

Add new cultural sources by:
1. **REST API**: `POST /api/v1/ingest/text` or `/ingest/file`
2. **Seeding script**: Edit `app/services/ingestion.py → ingest_sample_cultural_data()`
3. **Bulk ingestion**: Use `IngestionService` directly in a script

```python
from app.services.ingestion import get_ingestion_service

svc = get_ingestion_service()
svc.ingest_text(
    text=open("my_wiki_article.txt").read(),
    source="Wikipedia — Māori Culture",
    culture="Maori",
    language="en",
)
```

---

## Project Structure

```
culturarag/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── core/
│   │   └── config.py        # Pydantic settings
│   ├── models/
│   │   └── schemas.py       # Request/Response models
│   ├── api/
│   │   └── routes.py        # All API routers
│   ├── services/
│   │   ├── rag_chain.py     # FAISS → Prompt → Gemini pipeline
│   │   ├── vector_store.py  # FAISS index management
│   │   ├── ingestion.py     # Text chunking & document ingestion
│   │   └── feedback.py      # RLHF feedback collection
│   └── utils/
│       └── lora_finetune.py # LoRA/PEFT fine-tuning script
├── tests/
│   └── test_culturarag.py   # Pytest test suite
├── data/                    # Runtime data (gitignored)
│   ├── faiss_index/         # Persisted FAISS index
│   ├── feedback_log.jsonl   # RLHF feedback records
│   └── lora_adapters/       # Trained LoRA weights
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── run.py
```

---

## License

MIT
