# AnswerAI

AnswerAI is a FastAPI Retrieval-Augmented Generation service. It retrieves context with hybrid search, builds a prompt, and generates an answer with a local Llama model.

Core flow:

```text
POST /answer
  -> clean query
  -> retrieve context from Elasticsearch BM25 + Chroma dense search
  -> merge ranked document IDs
  -> build prompt
  -> generate answer with llama-cpp-python
  -> return answer, context, timing
```

## What this repo provides

- FastAPI inference API in `app/main.py`
- RAG orchestration in `app/rag.py`
- Hybrid retrieval, indexing, Redis cache, and prompt building in `app/storage.py`
- Local Llama and SentenceTransformer setup in `app/models.py`
- Runtime config through `.env`
- Local unit tests for retrieval correctness in `tests/test_storage.py`
- Windows-first setup guide in `SETUP_AND_TESTING.md`

## Requirements

For local tests:

- Python 3.10+
- PowerShell
- `requirements-dev.txt`

For the real API:

- Python 3.10+
- Docker Desktop
- Redis
- Elasticsearch
- local GGUF Llama model
- JSON dataset
- `requirements.txt`

## Quick start: clone, test, validate

```powershell
git clone <REPO_URL>
cd AnswerAI-main

py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

python -m pytest -q
python -m compileall app tests
```

Expected test result:

```text
4 passed
```

The unit tests do not require Redis, Elasticsearch, ChromaDB, an embedding model, or a Llama model. They use fakes and validate retrieval identity correctness.

## Run the real API locally

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create local config:

```powershell
Copy-Item .env.example .env
```

For a smoke test, keep:

```text
DATASET_PATH=examples/dataset.sample.json
```

Add a real local model:

```powershell
New-Item -ItemType Directory -Force app\Models
```

Copy your GGUF model to:

```text
app/Models/model.gguf
```

Start services:

```powershell
docker compose up -d redis elasticsearch
```

Run the API:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Test the endpoint from another PowerShell window:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/answer `
  -ContentType "application/json" `
  -Body '{"query":"What does AnswerAI use for retrieval?"}'
```

Expected response shape:

```json
{
  "answer": "...",
  "context": ["..."],
  "timing": {
    "retrieval": 0.0,
    "generation": 0.0,
    "total": 0.0
  }
}
```

## Configuration

Use `.env.example` as the source of truth:

```text
DATASET_PATH=examples/dataset.sample.json
DENSE_ENCODER_PATH=sentence-transformers/all-MiniLM-L6-v2
MODEL_PATH=app/Models/model.gguf
ES_HOST=http://localhost:9200
ES_INDEX=answerai-docs
REDIS_HOST=localhost
REDIS_PORT=6379
UVICORN_HOST=127.0.0.1
UVICORN_PORT=8000
```

Do not commit `.env`, model files, local datasets, `.venv`, `.pytest_cache`, `__pycache__`, or `chroma_db`.

## Dataset format

The expected dataset is a JSON list of documents:

```json
[
  {
    "id": "doc-a",
    "text": "Document text used for retrieval."
  }
]
```

Document IDs are treated as canonical strings across Elasticsearch, Chroma, and final context selection. They are not treated as list indexes.

## Project structure

```text
app/
  main.py       # FastAPI app and /answer endpoint
  rag.py        # query processing flow
  storage.py    # Redis, Elasticsearch, Chroma, prompt construction
  models.py     # SentenceTransformer and llama-cpp model pool
  config.py     # environment-driven configuration
  utils.py      # text cleaning, dataset loading, query logging
examples/
  dataset.sample.json
tests/
  test_storage.py
```

## Commit gate

Run this before committing:

```powershell
python -m pytest -q
python -m compileall app tests
```

Then review and commit:

```powershell
git status
git diff
git add .
git commit -m ""
```

## Current operational notes

- `docker-compose.yml` includes a Chroma service, but the current code uses local Chroma persistence through `persist_directory=chroma_db`.
- The first run may download the SentenceTransformer model if `DENSE_ENCODER_PATH` points to a Hugging Face model name.
- On Windows, `llama-cpp-python` may require Microsoft C++ Build Tools if no compatible wheel is available.
