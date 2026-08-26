# Umbrella — Production-Oriented RAG System

Umbrella is an open-source research and engineering project designed to study, build, and incrementally evaluate Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```
umbrella/
├── src/
│   ├── ingestion/       # Document loaders, cleaner, chunker, metadata tagger
│   ├── query/           # Retriever, context validation, prompt construction, generator
│   ├── chains/          # LCEL composition pipelines
│   ├── storage/         # Storage layer (ChromaDB vector store)
│   ├── eval/            # RAGAS evaluation runner and metrics store
│   ├── api/             # FastAPI entry point, routes, schemas, middleware
│   ├── config/          # Centralized configuration (pydantic-settings)
│   └── common/          # Structured logging and typed exceptions
├── frontend/            # Streamlit multi-page UI
├── tests/
│   ├── unit/            # Unit tests for pure components
│   └── integration/     # End-to-end pipeline tests
├── eval/
│   ├── datasets/        # Evaluation benchmark datasets
│   └── results/         # Permanent evaluation run artifacts
├── decision_log/        # Engineering decision records
└── data/
    ├── documents/       # Uploaded document storage
    └── chroma_db/       # ChromaDB vector store persistence
```

## Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and set your Groq API key:
```bash
cp .env.example .env
```

### 3. Run Backend (FastAPI)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Frontend (Streamlit)
```bash
streamlit run frontend/app.py
```