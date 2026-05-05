# CulturaRAG: AI Knowledge Explorer for World Cultures & Languages

CulturaRAG is a Retrieval-Augmented Generation (RAG) system designed to explore cultural knowledge across languages. It combines semantic search (FAISS), LLM-based generation (Google Gemini), and a feedback-driven improvement loop using RLHF and LoRA fine-tuning.

The goal is to provide **context-aware, multilingual answers** while continuously improving from user feedback.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                       │
│                                                          │
│  POST /query  →  RAG Pipeline                            │
│                     │                                    │
│                     ├─ FAISS Vector Search               │
│                     │   (multilingual embeddings)        │
│                     │                                    │
│                     └─ Gemini (via LangChain)            │
│                         context-based generation         │
│                                                          │
│  POST /ingest →  Text Chunking → FAISS Index             │
│                                                          │
│  POST /feedback → Feedback Log (JSONL)                   │
│                     │                                    │
│                     └─ LoRA Fine-tuning Pipeline         │
└──────────────────────────────────────────────────────────┘
```

### How it works

1. A query is sent to the API
2. Relevant documents are retrieved using FAISS
3. Retrieved context is passed to Gemini for response generation
4. Feedback is stored and later used for fine-tuning

---

## Tech Stack

* **Backend:** FastAPI + Uvicorn
* **LLM Orchestration:** LangChain
* **Model:** Google Gemini 1.5 Pro
* **Vector Store:** FAISS (cosine similarity)
* **Embeddings:** SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Fine-tuning:** PEFT / LoRA
* **Parsing:** PyPDF, python-docx
* **Config:** Pydantic + dotenv
* **Testing:** Pytest
* **Containerization:** Docker

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/culturarag.git
cd culturarag
```

---

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Configure environment variables

```bash
cp .env.example .env
```

Add your Gemini API key in the `.env` file.

---

### 4. Run the server

```bash
python run.py
```

* API: http://localhost:8000
* Docs: http://localhost:8000/docs

On first run, the system seeds the vector database with sample cultural data.

---

## API Usage

### Query the system

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the significance of sakura in Japanese culture?",
    "language": "en"
  }'
```

---

### Multilingual query

```json
{
  "question": "Explain the Ubuntu philosophy",
  "language": "es"
}
```

---

### Ingest new content

```bash
POST /api/v1/ingest/text
```

You can also upload documents using `/api/v1/ingest/file`.

---

### Submit feedback

```bash
POST /api/v1/feedback
```

Feedback is stored and later used to improve the system.

---

## Feedback Loop & Fine-tuning

The system includes a lightweight RLHF pipeline:

1. User feedback is recorded
2. Data is stored in `feedback_log.jsonl`
3. High-quality examples are extracted
4. LoRA fine-tuning is run offline

Example:

```bash
python -m app.utils.lora_finetune \
  --dataset ./data/feedback_log.jsonl \
  --base-model google/gemma-2b \
  --output-dir ./data/lora_adapters
```

This allows iterative improvement without retraining the full model.

---

## Project Structure

```
culturarag/
├── app/
│   ├── api/          # API routes
│   ├── services/     # RAG, ingestion, feedback logic
│   ├── models/       # Data schemas
│   ├── core/         # Configuration
│   └── utils/        # Fine-tuning scripts
├── data/             # Runtime data (gitignored)
├── tests/            # Test suite
├── docker-compose.yml
├── requirements.txt
└── run.py
```

---

## Running Tests

```bash
pytest tests/
```

---

## Docker Deployment

```bash
docker-compose up --build
```

The FAISS index and feedback logs are persisted in the `data/` directory.

---

## Future Improvements

* Improve multilingual retrieval quality
* Add evaluation metrics for responses
* Automate the fine-tuning workflow
* Add authentication and rate limiting
