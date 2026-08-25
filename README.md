# Umbrella: Production-Grade RAG System (V1)

Umbrella is a modular, production-oriented Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain (LCEL)**, **ChromaDB**, **Groq API**, and **Streamlit**.

The project emphasizes rigorous engineering principles: evidence-based design, hallucination defense, structured machine-verifiable citations, and transparent decision logging.

---

## 🏗 Architecture Overview

```
[Ingestion Pipeline - LCEL]
Upload (PDF/DOCX/TXT/MD)
  → Loader
  → Cleaner (Regex-based, pure functions)
  → Chunker (RecursiveCharacterTextSplitter)
  → Metadata Tagger (with SHA-256 idempotency)
  → Storage (ChromaDB with built-in embeddings)

[Query Pipeline - LCEL]
User Query
  → Retriever (top-k=3, similarity search)
  → Context Validator (relevance gate)
  → Prompt Builder (context + inline citation tags)
  → LLM Generation (Groq API, temperature=0.7)
  → Output Validator (citation existence & coverage)
  → Response (grounded answer + citations, or structured refusal)
```

---

## 📁 Repository Structure

```
umbrella/
├── src/
│   ├── ingestion/       # Format loaders, regex cleaner, text chunker, metadata builder
│   ├── query/           # Retriever, context validator, prompt builder, Groq generator, output validator
│   ├── chains/          # LCEL composition chains (ingestion_chain.py, query_chain.py)
│   ├── storage/         # ChromaDB integration (shared storage layer) & Postgres placeholder
│   ├── eval/            # RAGAS evaluation runner, dataset loader, metrics store
│   ├── api/             # FastAPI app, route handlers, Pydantic schemas, middleware
│   ├── config/          # pydantic-settings configuration
│   └── common/          # Structured JSON logging, typed error hierarchy
├── frontend/            # Streamlit 3-page web UI (Landing, Workspace, Evaluation)
├── tests/               # Unit and integration test suites
├── eval/                # Evaluation datasets and historical experiment results
├── decision_log/        # Problem → Experiment → Decision engineering logs
└── data/                # Document storage and persistent ChromaDB vectors
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Groq API Key (from [console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/SingRajat/umbrella.git
cd umbrella

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the sample environment file and configure your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.7
```

### 4. Running the Backend
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### 5. Running the Frontend
```bash
streamlit run frontend/app.py
```

### 6. Running Tests
```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v
```