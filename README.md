# Umbrella — Production-Grounded RAG System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-Qwen_3.8--27B-f55036)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-77%20Passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Umbrella is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for **factual grounding, citation traceability, and quantifiable evaluation**.

Instead of treating RAG as an unchecked prompt wrapper over vector similarity, Umbrella decouples ingestion and query execution into strict LangChain Expression Language (LCEL) chains governed by pre-generation confidence gates, post-generation citation validation, and structured refusal contracts.

---

## Architecture Overview

```mermaid
graph TD
    subgraph Ingestion [1. Ingestion Pipeline - LCEL]
        A[File Upload: PDF / DOCX / TXT / MD] --> B[Format Loaders]
        B --> C[Regex Sanitizer & Cleaner]
        C --> D[Recursive Chunker + Heading Extraction]
        D --> E[Metadata Tagger: doc_id, page, section]
        E --> F[(ChromaDB Vector Store)]
    end

    subgraph Query [2. Query Pipeline - LCEL]
        G[User Query] --> H[Top-K Similarity Retrieval]
        F -.-> H
        H --> I{Context Confidence Gate}
        I -- Score < 0.5 or Empty -->> J[Refusal Response: INSUFFICIENT_CONTEXT]
        I -- Score >= 0.5 -->> K[Numbered Prompt Assembly]
        K --> L[Groq LLM Generation]
        L --> M[Inline Citation & Output Validator]
        M --> N[Verified Cited Answer / SSE Stream]
    end

    subgraph Evaluation [3. Offline Evaluation Harness]
        O[50-Sample Golden Dev Set] --> P[Evaluation Runner]
        P --> Q[RAGAS Metric Scoring: Faithfulness, Recall, Precision]
        Q --> R[(Persisted Artifacts /eval/results)]
    end

    style Ingestion fill:#1e1e2e,stroke:#89b4fa,stroke-width:1px,color:#cdd6f4
    style Query fill:#1e1e2e,stroke:#a6e3a1,stroke-width:1px,color:#cdd6f4
    style Evaluation fill:#1e1e2e,stroke:#f9e2af,stroke-width:1px,color:#cdd6f4
```

---

## Core Guarantees & Hallucination Defense

1. **Pre-Generation Confidence Gate:** Retrieved chunks are scored against a similarity threshold (`similarity_threshold >= 0.5`). If evidence is missing or low-confidence, the system halts immediately and returns a structured refusal (`INSUFFICIENT_CONTEXT`) rather than generating ungrounded prose.
2. **Deterministic Citations:** Output claims must reference retrieved blocks using inline brackets `[1]`, `[2]`. The post-generation validator parses every citation index and maps it back to exact document name, page number, and section heading.
3. **Ghost Citation Defense:** If the LLM generates a citation index outside the retrieved set (e.g. referencing `[4]` when only 3 chunks were supplied), the validator strips or flags the ungrounded citation.
4. **Thread-Pool Offloading:** Synchronous I/O operations (vector retrieval, disk reads, Groq calls) are routed via FastAPI standard worker threads, preventing event-loop stalls under concurrent load.
5. **Hardened Ingestion Security:** Multi-layer defense-in-depth sanitizes filenames against CWE-22 path traversal attacks (`Path.name` + `is_relative_to`), enforces memory-capped 64 KB streaming uploads up to 25 MB (`413 Payload Too Large`), and protects metadata integrity.

---

## Benchmark Evaluation Results (ADR-001)

Umbrella measures every architectural change against an automated offline benchmark harness using **RAGAS** and a 50-sample frozen golden technical development set:

| Metric | Score / Value | Description |
| :--- | :---: | :--- |
| **Faithfulness** | **85.5%** (`0.8547`) | Generated answers are strictly derived from retrieved evidence (zero hallucination). |
| **Context Recall** | **86.3%** (`0.8632`) | Retrieved chunks capture all required ground-truth facts. |
| **Context Precision** | **84.7%** (`0.8467`) | Relevant chunks rank at the top with minimal noise dilution. |
| **Answered Rate** | **90.0%** (45 / 50) | Queries passing the confidence threshold. |
| **Refusal Rate** | **10.0%** (5 / 50) | Correctly triggered safety refusal contract (`INSUFFICIENT_CONTEXT`). |
| **P50 Latency (Median)** | **3.99 s** | Vector retrieval: **363 ms** (~5%) \| LLM generation: **3.41 s** (~95%). |
| **P90 Latency** | **11.12 s** | Bounded under load with linear backoff rate-limit handling. |

*(Evaluator Model: `openai/gpt-oss-120b` via RAGAS / Groq)*

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Groq Cloud API Key ([console.groq.com](https://console.groq.com/))

### 2. Installation
```bash
git clone https://github.com/SingRajat/umbrella.git
cd umbrella

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
EVAL_MODEL=openai/gpt-oss-120b
TEMPERATURE=0.7

CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=3
SIMILARITY_THRESHOLD=0.5

CHROMA_PERSIST_DIR=data/chroma_db
MAX_UPLOAD_SIZE_MB=25
RATE_LIMIT_RPM=60
TRUST_PROXY=false
STRICT_REFUSAL=true
LOG_LEVEL=INFO
```

### 4. Running the Application

**Start the FastAPI Backend:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
* API Root: `http://localhost:8000`
* Interactive OpenAPI Docs: `http://localhost:8000/docs`

**Start the Streamlit Frontend (in a new terminal):**
```bash
streamlit run frontend/app.py
```
* Web Dashboard: `http://localhost:8501`

---

## API Reference

| Method | Endpoint | Description | Key Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Liveness probe verifying ChromaDB connectivity | `200` |
| `POST` | `/api/v1/documents` | Upload and ingest document (`.pdf`, `.docx`, `.txt`, `.md`) | `201`, `400`, `413` |
| `GET` | `/api/v1/documents` | Paginated list of ingested documents | `200` |
| `GET` | `/api/v1/documents/{doc_id}` | Retrieve document metadata | `200`, `404` |
| `DELETE` | `/api/v1/documents/{doc_id}` | Cascade delete document file and ChromaDB vectors | `200`, `404` |
| `POST` | `/api/v1/query` | Grounded RAG query (JSON or Server-Sent Events stream) | `200`, `400`, `404` |

### Sample Grounded Response (`POST /api/v1/query`)
```json
{
  "status": "answered",
  "answer": "Docker containers share the host kernel while maintaining process isolation [1].",
  "citations": [
    {
      "chunk_id": "c32c82a0_c02",
      "doc_id": "c32c82a0-ef69-473f-b025-a5bee0075811",
      "doc_name": "docker_cheat_sheet.md",
      "page_number": null,
      "section_heading": "Containers",
      "text_excerpt": "Containers are to Virtual Machines as threads are to processes..."
    }
  ],
  "token_usage": { "prompt_tokens": 749, "completion_tokens": 82, "total_tokens": 831 }
}
```

### Sample Refusal Response (When Context Is Insufficient)
```json
{
  "status": "refused",
  "reason": "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this query.",
  "retrieved_chunk_ids": ["c32c82a0_c15", "c32c82a0_c16"],
  "query": "How do I bake a chocolate cake?"
}
```

---

## Automated Testing & Verification

The test suite covers format loaders, regex sanitization, recursive chunking, ChromaDB upsert/delete cascades, rate limiting, and security hardening:

```bash
# Run full unit & integration test suite
pytest -v
```

```text
======================= 77 passed, 2 warnings in 20.32s =======================
```

To re-run the offline evaluation benchmark:
```bash
python eval/run_baseline.py
```
*(Artifacts and metrics are automatically saved into `eval/results/`)*

---

## Project Structure

```
umbrella/
├── src/
│   ├── api/                 # FastAPI routes, schemas, and security middleware
│   ├── ingestion/           # Loaders, regex cleaners, chunker, metadata schemas
│   ├── query/               # Retrievers, context validator, prompts, output validator
│   ├── chains/              # Composable LCEL ingestion & query pipelines
│   ├── storage/             # ChromaDB vector store and PostgreSQL abstractions
│   ├── config/              # Central Pydantic settings management
│   └── common/              # Structured logging and typed domain exception hierarchy
├── frontend/                # Streamlit multi-page UI (Landing, Workspace, Evaluation)
├── eval/                    # Offline evaluation runner, golden dataset, benchmark artifacts
├── decision_log/            # Architecture Decision Records (ADR-001)
├── tests/                   # 77 unit & integration tests
├── requirements.txt         # Project dependencies
└── PRD.md                   # Product Requirements Document
```

---

## Roadmap (V2)

- [ ] **Hybrid Search:** Combine BM25 sparse keyword search with dense ChromaDB embeddings via Reciprocal Rank Fusion (RRF) to eliminate typo and CLI-alias retrieval blind spots.
- [ ] **Conversational Memory:** Stateful multi-turn chat management with query condensation (coreference resolution) and prompt token budgeting.
- [ ] **Cross-Encoder Reranking:** Integrate Cohere / BGE rerankers to improve Context Precision on complex queries.

---

## License

Distributed under the [MIT License](LICENSE).