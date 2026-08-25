### 1. Project architecture

-  Frontend: streamlit for now 
-  Backend: FastAPI? `yes` 
-  Database: Postgres or supabase
-  Vector DB: chromdb 
-  Cache: streamlit inbuilt/redis
-  Deployment: rendere and streamlit cloud 

### 2. RAG ingestion

What document types do you support?

-  PDF 
-  DOCX 
-  TXT 
-  Web pages 
-  .MD file

For ingestion:

-  How are documents parsed? -doucment loader?
-  Cleaning/regex steps? yes 
-  Chunking strategy currently? simple(we change later)
-  Embedding model? Chromadb 
-  Vector DB/index? Chromab
-  Metadata stored per chunk? yes(doc\_id,source,page,chunk\_id)

### 3. Retrieval

Tell me what you currently use:

-  Dense/vector retrieval: `y`es 
-  Hybrid retrieval: `no` 
-  BM25: `no` 
-  Reranker: `no` 
-  Query rewriting: `no` 
-  Top-K: 3
-  Similarity threshold: not implemented

### 4. Generation

-  LLM provider: Grok API 
-  Model: not decided 
-  Streaming response: `yes`
-  Maximum context: not implemented 
-  Temperature: not decided 
-  Prompt system: not implemented 

### 6. Users & sessions

This is important for API design.

-  Authentication required? `no` 
-  Login/signup? `no` 
-  JWT/API keys/OAuth? Not yet 
-  Conversation/session history? `yes`( we will use streamlit inbuilt features for this) 
-  Multiple users simultaneously? `yes`(lets try) 
-  User-specific document collections? Not decided

### 7. Core user operations

Tell me which operations your frontend needs:

```
no additions for now
```

```
Upload document
Delete document
List documents
Ask question
Stream answer
View citations
Create conversation
Get conversation history
Delete conversation
```

### 8. Evaluation harness

Since you're comparing:

```
Baseline
→ Semantic Chunking
→ Hybrid Retrieval
→ Reranker
→ Final RAG
```

tell me:

-  Ground-truth questions: `50–60` 
-  Where stored: Not decided 
-  Evaluation framework: RAGAS eval 
-  Metrics: faithfullness,answer correctiveness,precision,recall 
-  Do you want evaluation exposed through FastAPI or run offline? offline for now

9. Production requirements(not required for now but design in such way that this could be accomodated later)

### 10. Infrastructure(same as production requirements)

### 11. Current code

Most useful of all: give me your current project structure,(not created keep space for it)


# User requirement document

## Project: Umbrella — RAG pipeline with incremental improvements

### Introduction

Umbrella is an open-source research project dedicated to studying and incrementally improving Retrieval-Augmented Generation (RAG) systems. The project starts with a minimal working RAG and progressively adds components to compare their impact.

### Goals

- Create a baseline RAG from scratch
- Implement incremental improvements (semantic chunking → hybrid retrieval → reranking)
- Evaluate each step using RAGAS
- Allow easy comparison of different techniques
- Provide modular, extensible code

### Scope

- Local, single-machine operation (no multi-user / enterprise features initially)
- All components must be open-source and self-hostable
- Single user, single session mode is sufficient for baseline
- Evaluation is offline, not real-time during inference

### Architecture requirements

- Modular Python stack
- Clear separation between:
  - Document ingestion
  - Embedding
  - Indexing
  - Retrieval
  - Generation
  - Evaluation
- Frontend for experimentation
- Backend to orchestrate

### Functional requirements

#### Document ingestion

- Supported formats:
  - PDF
  - DOCX
  - TXT
  - Markdown (`.md`)
  - Web pages (URLs)
- Processing pipeline:
  - Remove noise (headers/footers, page numbers, artifacts)
  - Clean text using regex
  - Chunking (default strategy → semantic chunking)
- Metadata storage per chunk:
  - `doc_id`
  - `source` (filename or URL)
  - `page` (for PDFs)
  - `chunk_id`

#### Embedding & Indexing

- Embedding model:
  - Sentence-Transformers (default open-source model)
- Vector DB:
  - ChromaDB
- Indexing:
  - `id`
  - `vector`
  - `metadata`

#### Retrieval

- Dense retrieval (vector search)
- Support for hybrid search (optional later)
- Support for reranking (optional later)
- Configurable Top-K
- Similarity threshold

#### Generation

- LLM provider:
  - Grok API (primary)
  - Fallback to Mistral/Llama (open-source models)
- Streaming responses
- Configurable temperature
- Context window handling
- Prompt templates:
  - System prompt
  - User prompt
  - Few-shot examples (optional)

#### Conversation & history

- Session-per-user (local)
- Conversation history storage
- Ability to chat over previous turns
- Export/resume conversations

#### Evaluation

- Ground-truth questions
- RAGAS framework
- Metrics:
  - Faithfulness
  - Answer correctness
  - Answer relevancy
  - Context precision
  - Context recall
- Incremental comparison:
  - Baseline only
  - + Semantic chunking
  - + Hybrid retrieval
  - + Reranker
  - Final complete system

### Non-functional requirements

- Easy to run locally
- Clear README for each module
- Modular, pluggable components
- Environment-variable configuration
- Logging for ingestion, retrieval, generation, evaluation
- Minimal external dependencies
- All code open-source, permissively licensed

### User interface

A simple web UI for experimentation:

- Upload documents
- List documents
- Ask questions
- View answers with citations
- Switch between RAG configurations
- View evaluation results
- Export/import conversation

### Incremental improvement steps

1. Baseline RAG

   - Standard chunking
   - Dense retrieval
   - Grok API

2. Semantic chunking

   - Apply semantic-based chunking
   - Compare with baseline

3. Hybrid retrieval

   - Add BM25
   - Combine dense + sparse retrieval

4. Reranker

   - Add cross-encoder reranker
   - Improve answer relevance

5. Production optimizations

   - Conversation history
   - Multiple users support (multi-tenant ready)
   - API versioning

### Evaluation requirements

- Evaluation dataset of 50–60 questions
- Ground-truth answers for each question
- Results stored per configuration
- Comparison tables & charts

### Technical constraints

- No vendor lock-in where possible
- Open-source alternatives should be available
- Local execution is mandatory for baseline

### Project structure (proposed)

```
umbrella/
├── src/
│   ├── ingestion/
│   ├── embedding/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
├── frontend/
├── tests/
├── experiments/
├── config/
├── data/
│   ├── documents/
│   └── evaluation/
├── notebooks/
└── README.md
```

### Success criteria

- A working RAG system can be run locally
- Each improvement step shows measurable impact
- Evaluation is reproducible
- Code is modular and easy to extend
- Documentation is clear and complete

->These are the features of my project, infer to these features to design api for my project(ignore the API Section given in PRD.md)
->For every API specify:

- Purpose
- Request flow
- Response structure
- Error handling
- Caching strategy
- Retry strategy


## Rules
Cross-cutting API requirements:
- **Validation:** every request body validated against a Pydantic schema before touching pipeline code; malformed requests never reach ingestion/query logic.
- **Rate limiting:** token-bucket, per-API-key (or per-IP for the unauthenticated demo case) — justified because both embedding and LLM calls are metered, paid resources; without this, a single misbehaving client can exhaust budget.
- **Error format:** consistent `{error_code, message, correlation_id}` across all endpoints.
- **Correlation IDs:** every request gets one, propagated through logs across ingestion/query/eval so a single request's full trace can be reconstructed for debugging.
- **Caching strategy per endpoint:** `/query` benefits from the query cache (§9) when enabled; `/documents` (GET) is cheap and uncached; `/eval/results` results are immutable once written and can be cached indefinitely by `run_id`.
- **Retry strategy:** outbound calls to the embedding/LLM provider use bounded exponential backoff (assumption: 3 attempts, base 1s) on transient errors (5xx/timeout) only — not on 4xx (bad request) errors, which should fail fast.