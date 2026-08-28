# Umbrella — Production-Oriented Grounded RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-Qwen_3.8--27B-f55036)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-50%20Passing-brightgreen?logo=pytest&logoColor=white)](file:///tests)

Umbrella is a modular, evidence-backed Retrieval-Augmented Generation (RAG) system engineered for **measurable evaluation, hallucination defense, and verifiable source citations**.

Instead of treating RAG as a simple prompt wrapper around a vector search, Umbrella implements a decoupled, two-pipeline architecture (Ingestion and Query) governed by strict context gates, inline citation extraction, citation-existence validation, and a standardized refusal contract.

---

## Why Umbrella?

Standard LLM chatbots and naive RAG implementations suffer from four critical production failure modes:

1. **Hallucination & Sycophancy**: Generating plausible-sounding facts when retrieved context is irrelevant or empty.
2. **Untraceable Answers**: Producing text without machine-verifiable references to source passages.
3. **Over-Retrieval Noise**: Forcing low-confidence chunks into the prompt, diluting LLM attention.
4. **Vibe-Based Engineering**: Tweaking chunk sizes, retrievers, or prompts based on intuition rather than quantifiable evaluation.

**Umbrella solves this by:**
- **Context Confidence Gating**: Evaluating retrieved chunks against a similarity threshold (`similarity_threshold >= 0.5`) before calling the LLM.
- **Strict Grounding Contract**: Instructing the model to answer *only* from numbered context blocks and explicitly refuse if evidence is absent (`INSUFFICIENT_CONTEXT`).
- **Machine-Verifiable Citations**: Tracing every claim back to document, page, and section metadata.
- **Evaluation Discipline**: Building against a reproducible offline evaluation harness (RAGAS) where every architectural change is benchmarked.

---

## System Architecture

Umbrella decouples document processing and question answering into two independent LangChain Expression Language (LCEL) chains connected solely via the vector storage layer.

```mermaid
graph TD
    subgraph Ingestion Pipeline [1. Ingestion Pipeline - LCEL]
        A[Document: PDF / DOCX / TXT / MD] --> B[Format-Specific Loader]
        B --> C[Regex Cleaner & Sanitizer]
        C --> D[Recursive Chunker + Heading Extraction]
        D --> E[Metadata Tagger: doc_id, page, section]
        E --> F[(ChromaDB Vector Store)]
    end

    subgraph Query Pipeline [2. Query Pipeline - LCEL]
        G[User Query] --> H[Top-K Similarity Retrieval]
        F -.-> H
        H --> I{Context Validation Gate}
        I -- Score < 0.5 or Empty -->> J[Refusal Response: INSUFFICIENT_CONTEXT]
        I -- Score >= 0.5 -->> K[Numbered Prompt Assembly]
        K --> L[ChatGroq LLM Generation]
        L --> M[Output & Citation Validator]
        M --> N[Cited Answer Response / SSE Stream]
    end

    subgraph Evaluation Harness [3. Offline Evaluation Harness]
        O[Golden QA Dataset] --> P[Evaluation Runner]
        P --> Q[Query Pipeline Execution]
        Q --> R[RAGAS Metrics: Faithfulness, Recall, Precision]
        R --> S[(Stored Run Artifacts /eval/results)]
    end

    style Ingestion Pipeline fill:#1e1e2e,stroke:#89b4fa,stroke-width:1px,color:#cdd6f4
    style Query Pipeline fill:#1e1e2e,stroke:#a6e3a1,stroke-width:1px,color:#cdd6f4
    style Evaluation Harness fill:#1e1e2e,stroke:#f9e2af,stroke-width:1px,color:#cdd6f4
```

---

## End-to-End Workflow

### 1. Ingestion Flow
1. **Upload**: User uploads a `.pdf`, `.docx`, `.txt`, or `.md` file via the REST API (`POST /api/v1/documents`) or Streamlit Workspace.
2. **Extraction**: Format-specific loader ([`loaders.py`](file:///src/ingestion/loaders.py)) parses raw text while tracking source-level metadata (page numbers, document name, SHA-256 hash).
3. **Cleaning**: Deterministic regex rules ([`cleaner.py`](file:///src/ingestion/cleaner.py)) strip recurring cross-page headers/footers, remove non-printable control characters, de-hyphenate broken words, and normalize whitespace.
4. **Chunking**: Recursive character chunker ([`chunker.py`](file:///src/ingestion/chunker.py)) splits text with configurable overlap (`chunk_size=800`, `chunk_overlap=100`) while preserving section headings.
5. **Metadata Construction**: Each chunk is wrapped into a typed [`ChunkRecord`](file:///src/ingestion/metadata.py) with unique `chunk_id`, `doc_id`, `doc_name`, `page_number`, and `section_heading`.
6. **Vector Indexing**: Chunks are embedded and indexed into persistent ChromaDB storage ([`chroma.py`](file:///src/storage/chroma.py)).

### 2. Query & Generation Flow
1. **Query Submission**: User submits a question via `POST /api/v1/query` or Server-Sent Events stream (`POST /api/v1/query/stream`).
2. **Vector Retrieval**: ChromaDB performs top-$k$ similarity search ($k=3$), mapping distance metrics to cosine similarity scores ($1 - \text{distance}$).
3. **Context Validation**: Gating function ([`context_validator.py`](file:///src/query/context_validator.py)) verifies that retrieved evidence passes the confidence threshold (`similarity_threshold >= 0.5`).
   - If confidence is low or no chunks match $\rightarrow$ immediate structured refusal without wasting LLM tokens.
4. **Prompt Assembly**: Valid chunks are formatted into numbered blocks `[1]`, `[2]` with document and section tags ([`prompt.py`](file:///src/query/prompt.py)).
5. **LLM Generation**: Prompt is dispatched to Groq API ([`generator.py`](file:///src/query/generator.py)) using `qwen/qwen3.8-27b` with bounded exponential backoff retries.
6. **Output & Citation Validation**: Post-generation validator ([`output_validator.py`](file:///src/query/output_validator.py)) extracts inline citation indices (e.g., `[1]`, `[2]`), cross-checks them against the retrieved set, and formats verified [`CitationItem`](file:///src/api/schemas.py) objects.

---

## Citation Architecture

Citations in Umbrella are **deterministic and machine-verifiable**, not ungrounded prose.

```
Document Upload
   │
   ├── Page 2 ──► Chunk ID: "a1b2...#c0" ──► Metadata: {doc_id, filename, page: 2, section: "Overview"}
   │
Vector Search & Context Injection
   │
   └── Prompt Block: "[1] (Document: report.pdf, Page: 2, Section: Overview)\n..."
   │
LLM Generation & Extraction
   │
   ├── LLM Output: "Revenue grew 14% year-over-year [1]."
   │
Resolution
   └── Citation Item: {chunk_id, doc_name: "report.pdf", page_number: 2, section_heading: "Overview", text_excerpt: "..."}
```

- **Chunk Metadata Preservation**: Metadata remains attached across extraction $\rightarrow$ chunking $\rightarrow$ storage $\rightarrow$ prompt injection $\rightarrow$ response.
- **Hallucinated Citation Defense**: If the LLM generates a citation index outside the retrieved set (e.g., citing `[4]` when only 3 chunks were provided), the output validator detects and strips the invalid reference.
- **Transparent Refusals**: When the system refuses to answer due to insufficient context, the API returns the retrieved chunk IDs alongside the refusal reason so developers can inspect retrieval quality.

---

## Key Features

- **Multi-Format Document Ingestion**: Native parsing for PDF (`pypdf`, `pdfplumber`), DOCX (`python-docx`), plain text (`.txt`), and Markdown (`.md`).
- **Deterministic Cleaning Pipeline**: Pure regex functions for cross-page recurring header/footer removal, dehyphenation, and control character sanitization.
- **Confidence Gate / Context Validator**: Configurable similarity thresholding before LLM invocation.
- **Dual Query Protocols**: Synchronous JSON endpoint (`/api/v1/query`) and real-time Server-Sent Events (SSE) streaming (`/api/v1/query/stream`).
- **Structured Error Handling**: Custom typed exceptions (`IngestionError`, `RetrievalError`, `GenerationError`, `ValidationError`) mapped to HTTP status codes with correlation ID tracking.
- **Production Rate Limiting & Security**: Window-based rate limiting (`rate_limit_rpm=60`), max upload constraints (`25 MB`), and OWASP security headers.
- **Interactive Multi-Page Frontend**: Streamlit UI featuring Landing Page, Document Ingestion & Registry, Streaming Chat with expandable citation cards, and Evaluation status views.

---

## Engineering Decisions & Trade-offs

| Component | Decision | Rationale | Trade-off / Alternative Considered |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | Native async support, automatic OpenAPI/Swagger documentation, strict Pydantic type validation. | Heavier setup than Flask; preferred over Django for microservice modularity. |
| **Frontend UI** | **Streamlit** | Rapid iterative prototyping, native session state, zero-overhead multi-page routing. | Less layout customization than React/Next.js; ideal for internal tools and engineering demos. |
| **Vector Store** | **ChromaDB** | Serverless local persistence (`sqlite3` + parquet), zero external infrastructure dependencies, fast local testing. | Single-node only; planned migration to PostgreSQL + `pgvector` for multi-tenant scale. |
| **Orchestration** | **LangChain LCEL** | Declarative pipeline composition (`RunnableLambda`), isolated unit-testability of stages, clean swapability of components. | Requires adhering to LCEL input/output schema contracts; avoids monolithic procedural code. |
| **LLM Provider** | **Groq (`qwen/qwen3.8-27b`)** | Extremely low inference latency (sub-second time-to-first-token), strong instruction-following and citation compliance. | External API dependency; mitigated by bounded retries and exponential backoff. |
| **Chunking Strategy** | **Fixed-size + Overlap** | Simple, deterministic baseline (`chunk_size=800`, `chunk_overlap=100`) preserving section boundaries. | May split mid-sentence across complex tables; semantic/hierarchical chunking slated for V2 comparison. |
| **Context Gate** | **Pre-generation Validation** | Drops retrieval noise and halts pipeline execution if similarity $< 0.5$. | May refuse borderline queries; prevents hallucinations and saves LLM inference cost. |

---

## Tech Stack

- **Core & Runtime**: Python 3.11+, Pydantic v2, `pydantic-settings`
- **Backend API**: FastAPI, Uvicorn, Starlette Middleware
- **RAG & Orchestration**: LangChain Core (LCEL), `langchain-groq`, `langchain-chroma`
- **Vector Storage**: ChromaDB
- **LLM**: Groq Cloud API (`qwen/qwen3.8-27b`)
- **Document Processing**: `pypdf`, `pdfplumber`, `python-docx`
- **Frontend**: Streamlit
- **Testing & Quality**: Pytest, HTTPX TestClient

---

## Project Structure

```
umbrella/
├── src/
│   ├── api/                 # FastAPI application, routes, schemas, middleware
│   │   ├── main.py          # App initialization, CORS, middleware registration
│   │   ├── routes.py        # /health, /documents, /query, /query/stream
│   │   ├── schemas.py       # Pydantic request/response data contracts
│   │   └── middleware.py    # Request correlation IDs, timing, rate limiting
│   ├── ingestion/           # Document ingestion components
│   │   ├── loaders.py       # PDF, DOCX, TXT, MD file loaders
│   │   ├── cleaner.py       # Pure regex text cleaning & header stripping
│   │   ├── chunker.py       # Recursive chunker with heading extraction
│   │   └── metadata.py      # ChunkRecord schema & ID generator
│   ├── query/               # Retrieval and generation components
│   │   ├── retriever.py     # ChromaDB vector retrieval & score mapping
│   │   ├── context_validator.py # Relevance threshold gate (threshold >= 0.5)
│   │   ├── prompt.py        # System prompt & numbered context formatter
│   │   ├── generator.py     # ChatGroq client with retry logic
│   │   └── output_validator.py # Inline citation parser & refusal detector
│   ├── chains/              # Composable LCEL execution pipelines
│   │   ├── ingestion_chain.py # load -> clean -> chunk -> store
│   │   └── query_chain.py   # retrieve -> validate -> prompt -> generate -> parse
│   ├── storage/             # Vector and relational storage layer
│   │   ├── chroma.py        # ChromaVectorStore wrapper
│   │   └── postgres.py      # PostgreSQL persistence foundation
│   ├── config/              # Central configuration
│   │   └── settings.py      # Pydantic BaseSettings loading .env
│   └── common/              # Shared infrastructure
│       ├── logging.py       # Structured logger with request context
│       └── errors.py        # Typed exception hierarchy
├── frontend/                # Streamlit multi-page UI
│   ├── app.py               # Main navigation & backend health monitor
│   ├── components/          # Reusable UI widgets (chat, upload, citations)
│   └── pages/               # 1_Landing.py, 2_Workspace.py, 3_Evaluation.py
├── eval/                    # Evaluation harness
│   ├── datasets/            # Evaluation benchmark datasets
│   └── results/             # Persisted evaluation run artifacts
├── decision_log/            # Engineering decision records
├── data/                    # Persistent storage
│   ├── documents/           # Uploaded files & registry.json
│   └── chroma_db/           # ChromaDB vector database files
├── tests/                   # Automated test suite
│   ├── unit/                # 49 unit tests for pure functions & endpoints
│   └── integration/         # Integration test for end-to-end pipelines
├── .env.example             # Environment configuration template
├── requirements.txt         # Production and development dependencies
├── Tracker.md               # Phase & milestone progression tracker
└── PRD.md                   # Product requirements document
```

---

## API Overview

Interactive Swagger documentation is available at `http://localhost:8000/docs` when running the backend.

### 1. Health Probe
- **`GET /api/v1/health`**
- **Response**: `200 OK`
  ```json
  {
    "status": "healthy",
    "chromadb": "connected",
    "version": "0.1.0"
  }
  ```

### 2. Document Ingestion
- **`POST /api/v1/documents`** (Multipart Form: `file=@document.pdf`)
- **Response**: `201 Created`
  ```json
  {
    "doc_id": "709f2760-c98d-40d7-8088-d0ab1149e7b4",
    "filename": "Transformers.pdf",
    "status": "ingested",
    "chunk_count": 42,
    "source_type": "pdf",
    "ingested_at": "2026-08-28T02:55:00Z"
  }
  ```

### 3. List Ingested Documents
- **`GET /api/v1/documents?page=1&page_size=10`**
- **Response**: `200 OK`
  ```json
  {
    "documents": [
      {
        "doc_id": "709f2760-c98d-40d7-8088-d0ab1149e7b4",
        "filename": "Transformers.pdf",
        "source_type": "pdf",
        "chunk_count": 42,
        "ingested_at": "2026-08-28T02:55:00Z"
      }
    ],
    "pagination": { "total": 1, "page": 1, "page_size": 10, "has_next": false }
  }
  ```

### 4. Delete Document
- **`DELETE /api/v1/documents/{doc_id}`**
- **Response**: `200 OK`
  ```json
  {
    "doc_id": "709f2760-c98d-40d7-8088-d0ab1149e7b4",
    "status": "deleted",
    "chunks_removed": 42
  }
  ```

### 5. Grounded RAG Query
- **`POST /api/v1/query`**
- **Request**:
  ```json
  {
    "query": "What is the primary advantage of the self-attention mechanism?",
    "doc_id": "709f2760-c98d-40d7-8088-d0ab1149e7b4"
  }
  ```
- **Response (Answered)**: `200 OK`
  ```json
  {
    "status": "answered",
    "answer": "Self-attention allows the model to connect all positions with a constant number of sequentially executed operations [1].",
    "citations": [
      {
        "chunk_id": "709f2760_c12",
        "doc_id": "709f2760-c98d-40d7-8088-d0ab1149e7b4",
        "doc_name": "Transformers.pdf",
        "page_number": 6,
        "section_heading": "Why Self-Attention",
        "text_excerpt": "Self-attention, sometimes called intra-attention..."
      }
    ]
  }
  ```
- **Response (Refused)**: `200 OK`
  ```json
  {
    "status": "refused",
    "reason": "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information.",
    "retrieved_chunk_ids": ["709f2760_c01", "709f2760_c02"],
    "query": "What is the recipe for chocolate cake?"
  }
  ```

### 6. Streaming RAG Query
- **`POST /api/v1/query/stream`**
- **Response**: `200 OK` (`text/event-stream`)
  ```text
  data: {"type": "token", "content": "Self"}
  data: {"type": "token", "content": "-attention"}
  data: {"type": "token", "content": " connects all positions [1]."}
  data: {"type": "citations", "citations": [...]}
  data: {"type": "done"}
  ```

---

## Setup & Local Installation

### Prerequisites
- Python 3.11 or higher
- Groq API Key ([console.groq.com](https://console.groq.com/))

### 1. Clone & Setup Environment
```bash
git clone https://github.com/SingRajat/umbrella.git
cd umbrella

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and provide your credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL=qwen/qwen3.8-27b
TEMPERATURE=0.7

CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=3
SIMILARITY_THRESHOLD=0.5

CHROMA_PERSIST_DIR=data/chroma_db
MAX_UPLOAD_SIZE_MB=25
RATE_LIMIT_RPM=60
STRICT_REFUSAL=true
LOG_LEVEL=INFO
```

---

## Running Locally

### Start Backend Server (FastAPI)
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API available at `http://localhost:8000` | Documentation at `http://localhost:8000/docs`.

### Start Frontend UI (Streamlit)
In a separate terminal:
```powershell
streamlit run frontend/app.py
```
UI available at `http://localhost:8501`.

---

## Testing

The test suite contains **50 automated unit and integration tests** validating loader parsing, regex sanitization, chunk boundary preservation, ChromaDB indexing, API endpoints, rate limiting, and query execution.

Run the test suite:
```bash
pytest -v
```

```text
======================= 50 passed, 2 warnings in 17.04s =======================
```

---

## Evaluation Harness *(In Progress)*

Umbrella's evaluation framework is designed around **RAGAS** (Retrieval Augmented Generation Assessment) to prevent subjective optimization.

- **Target Metrics**:
  - **Faithfulness**: Proportion of claims in the generated answer directly supported by retrieved context.
  - **Context Precision**: Ratio of relevant retrieved chunks to total retrieved chunks.
  - **Context Recall**: Extent to which retrieved context covers the ground-truth answer.
  - **Latency & Token Efficiency**: Per-query TTFT, end-to-end latency, and token consumption.
- **Status**: Baseline dataset acquisition and runner configuration are currently in development under the `V1 — Evaluation` milestone (see [Tracker.md](file:///Tracker.md)). *Benchmark results will be published upon completion of the initial baseline run.*

---

## Deployment

### Backend (Render)
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/api/v1/health`

### Frontend (Streamlit Community Cloud)
- **Repository**: `SingRajat/umbrella`
- **Main file path**: `frontend/app.py`
- **Secrets**:
  ```toml
  BACKEND_URL = "https://your-backend-service.onrender.com"
  ```

---

## Limitations & Future Work

### Current Limitations (V1 Scope)
- **Fixed-size chunking**: May split complex tabular structures or multi-level legal clauses mid-context.
- **Single-node vector storage**: ChromaDB operates in local embedded mode; does not scale horizontally across multiple worker nodes.
- **Synchronous Ingestion**: File uploads execute synchronously without a distributed task queue (e.g., Celery/Redis).

### Planned Improvements (V2 Roadmap)
- [ ] **Semantic & Hierarchical Chunking**: Benchmark semantic splitting against fixed-size baseline.
- [ ] **Re-ranking Stage**: Integrate Cross-Encoder / Cohere reranker to evaluate Context Precision deltas.
- [ ] **Hybrid Search**: Combine BM25 sparse keyword search with dense ChromaDB embeddings.
- [ ] **PostgreSQL Storage Layer**: Transition document registry and chunk embeddings to `pgvector`.
- [ ] **Automated CI/CD Evaluation Gate**: Block pull request merges if RAGAS faithfulness drops below defined thresholds.

---

## Engineering Philosophy

Umbrella is built under a strict hypothesis-driven engineering methodology:

$$\text{Baseline Measurement} \longrightarrow \text{Identify Bottleneck} \longrightarrow \text{Isolated Architectural Change} \longrightarrow \text{Re-evaluate} \longrightarrow \text{Document Trade-off}$$

Every architectural change (chunk size, model selection, reranking, caching) must demonstrate a measurable improvement in accuracy, latency, or cost before being merged.

---

## License

Distributed under the [MIT License](file:///LICENSE).