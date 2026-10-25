# Umbrella

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

Evidence-backed retrieval-augmented generation service with citation verification and confidence-based refusal.

## Overview
Umbrella is a retrieval-augmented generation (RAG) system that enforces factual grounding across document ingestion and question answering. Standard RAG pipelines frequently output hallucinated claims or fail silently when retrieved context is irrelevant. Umbrella addresses this by gating retrieval behind a similarity score threshold, assembling numbered context blocks, and validating all inline citations against retrieved text before returning a response.

## Features
- Multi-format ingestion for PDF, DOCX, TXT, and Markdown files.
- Regex-based text cleaning and heading-preserving recursive chunking.
- Confidence-gated vector retrieval with automated refusal on low similarity.
- Post-generation citation validation mapping inline references to document metadata.
- FastAPI backend providing both standard JSON responses and Server-Sent Events (SSE) streaming.
- Streamlit web interface for document uploads and interactive querying.

## Tech Stack
- **Language**: Python 3.11+
- **Backend & Web**: FastAPI, Uvicorn, Streamlit
- **RAG & Storage**: LangChain, ChromaDB
- **LLM Provider**: Groq API (`qwen/qwen3.8-27b`)
- **Testing**: Pytest

## Architecture
```
[Document Upload] -> Format Loader -> Regex Cleaner -> Heading Chunker -> ChromaDB
                                                                             |
[User Query] -> Top-K Retrieval -> Similarity Gate (>= 0.5) -----------------+
                                         |
                       +-----------------+-----------------+
                       | (Pass)                            | (Fail)
                       v                                   v
             Prompt Assembly (Numbered)             Structured Refusal
                       |
                   Groq LLM
                       |
               Citation Validator -> Verified Answer
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- A Groq API key

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/SingRajat/umbrella.git
cd umbrella
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Configure the environment variables in `.env`:
```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=qwen/qwen3.8-27b
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=3
SIMILARITY_THRESHOLD=0.5
CHROMA_PERSIST_DIR=data/chroma_db
MAX_UPLOAD_SIZE_MB=25
RATE_LIMIT_RPM=60
```

### Running the Services
Start the FastAPI backend:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Streamlit frontend in a separate terminal:
```bash
streamlit run frontend/app.py
```

Run the test suite:
```bash
pytest -v
```

## Usage

### Query Endpoint
Submit a question to the REST API:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Where can Dockerized apps run?", "stream": false}'
```

Example response:
```json
{
  "status": "answered",
  "answer": "Dockerized apps can run on OS X and Windows laptops, QA servers running Ubuntu, and production VMs [1].",
  "citations": [
    {
      "chunk_id": "chunk_123",
      "doc_id": "doc_456",
      "doc_name": "docker_cheat_sheet.md",
      "page_number": null,
      "section_heading": "Why Docker",
      "text_excerpt": "Dockerized apps are completely portable..."
    }
  ],
  "token_usage": {
    "prompt_tokens": 749,
    "completion_tokens": 133,
    "total_tokens": 882
  }
}
```

## Project Structure
```text
├── data/                  # Storage for uploaded files and ChromaDB indices
├── eval/                  # Offline evaluation harness and benchmark datasets
├── frontend/              # Streamlit interface (app.py, components, pages)
├── src/
│   ├── api/               # FastAPI routes, schemas, and security middleware
│   ├── chains/            # Ingestion and query orchestration chains
│   ├── common/            # Error types and logging configuration
│   ├── config/            # Pydantic settings management
│   ├── ingestion/         # Document loaders, cleaners, and chunkers
│   ├── query/             # Context gates, prompt builders, and citation validators
│   └── storage/           # ChromaDB client and vector operations
└── tests/                 # Unit and integration test suites
```

## Limitations and Roadmap
- **Limitations**: Ingestion runs synchronously in thread pools rather than an asynchronous task queue; retrieval uses dense embeddings only.
- **Roadmap**:
  - Hybrid search combining BM25 sparse keyword matching with dense embeddings.
  - Multi-turn conversational memory with query coreference resolution.
  - Cross-encoder reranking.

## License and Contact
- **License**: TODO: <Specify license, e.g., MIT, and add a LICENSE file if intended>
- **Author**: Rajat Singh
- **GitHub**: https://github.com/SingRajat
- **Contact**: TODO: <Add email or LinkedIn profile URL>