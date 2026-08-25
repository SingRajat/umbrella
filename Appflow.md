AppFlow.md --- Production-Grade RAG Application

1. Product & Architecture

Goal

Build a production-oriented RAG application that lets a user upload a
document and ask questions about it. The application must make the RAG
pipeline measurable, explainable, citation-aware, and modular so that
later versions can be improved based on evaluation results.

V1 scope: establish a strong baseline RAG system before introducing
V2/V3 features.

High-Level Architecture

User
  │
  ▼
Streamlit Frontend
  │  HTTP / streaming requests
  ▼
FastAPI Backend
  │
  ├── Document Ingestion
  │     ├── Parse PDF / DOCX / TXT
  │     ├── Clean & preprocess
  │     ├── Chunk
  │     ├── Generate embeddings
  │     └── Store chunks + metadata in ChromaDB
  │
  ├── Retrieval
  │     ├── Embed query
  │     ├── Retrieve top-k (V1: k=3)
  │     └── Validate retrieved context
  │
  ├── Generation
  │     ├── Build citation-aware prompt
  │     ├── Generate answer
  │     └── Validate answer / citations
  │
  └── Evaluation / telemetry hooks

The frontend must not directly access the vector database, embedding
provider, or LLM provider. The FastAPI backend owns application logic
and external-service communication.

API.md is the authoritative source for endpoint contracts. Do not
invent endpoint names or payloads that conflict with API.md.

2. Application Pages & Navigation

Page 1 --- Landing Page

Purpose

Introduce the product and immediately communicate the core value: ask
questions about your documents and receive evidence-backed answers.

UI Sections

Hero - Product name - One-line value proposition - Short explanation
of citation-aware RAG - Primary CTA: Upload Document - Secondary CTA:
Learn How It Works

How It Works Show the V1 pipeline visually:

Upload → Clean → Chunk → Embed → Retrieve → Validate → Generate → Cite

Key Capabilities - PDF / DOCX / TXT support - Citation-aware
answers - Multi-source citations - Refusal when evidence is
insufficient - RAG evaluation with measurable metrics

Engineering / Trust section Communicate that the system is built
around: - Retrieval quality - Faithfulness - Context precision /
recall - Latency - Cost - Reliability

Interaction

Upload Document navigates to the workspace/upload flow.

Learn How It Works scrolls to the pipeline explanation; it should not
require a backend request.

Page 2 --- RAG Workspace

Purpose

Primary application screen where the user uploads a document and asks
questions.

Layout

Left / top: Document panel - Upload control - Supported file types -
File name - Processing state - Chunk count after successful ingestion -
Ingestion error state

Main: Chat panel - Question input - Submit action - Streaming answer
area where supported - Citation markers attached to claims -
Source/citation panel - Refusal state when evidence is insufficient

Optional metadata/source panel For each citation show: - Document
name - Page number when available - Chunk ID - Relevant source excerpt -
Source metadata

Upload Flow

User selects file
→ Frontend validates file type/basic constraints
→ Frontend sends multipart upload to FastAPI
→ Backend parses document
→ Backend cleans text
→ Backend chunks text
→ Backend generates embeddings
→ Backend stores vectors + metadata in ChromaDB
→ Backend returns ingestion status
→ Frontend updates document state
→ User can ask questions

The frontend should not assume ingestion succeeded until the backend
confirms success.

Query Flow

User submits question
→ Frontend sends query + document/session context
→ FastAPI validates request
→ Query embedding is generated
→ ChromaDB retrieves top 3 chunks
→ Retrieved context is validated
→ If evidence is insufficient:
      return citation-aware refusal
  Otherwise:
      build grounded prompt
      generate answer
      validate answer/citations
      stream/return response
→ Frontend renders answer + citations

The answer must be grounded in retrieved context. Unsupported claims
must not be presented as factual answers.

Page 3 --- Evaluation / Engineering Dashboard

Purpose

Show measurable system performance and make the project demonstrate
engineering decision-making rather than only chatbot functionality.

V1 Metrics

RAG quality - Faithfulness - Context recall - Context precision -
Answer relevance where available

System performance - Retrieval latency - Generation latency -
End-to-end latency - Input/output token usage - Failure/refusal rate

Experiment comparison Display baseline vs improved versions when
experiments are introduced.

Example:

Metric                 Baseline       Experiment       Delta
Faithfulness             X%              Y%            +Z pp
Context Precision        X%              Y%            +Z pp
Context Recall           X%              Y%            +Z pp
E2E Latency              X ms            Y ms          -Z%

Do not display fabricated values. Until evaluation is actually run, show
the metric as unavailable or pending.

Decision Log

Each major change should capture:

Problem → Hypothesis → Experiment → Decision → Trade-off → Measured Result

Examples: - Chunk size - Chunk overlap - Embedding model - Retrieval
k - Metadata filtering - Caching - Concurrency - Validation strategy

3. Server ↔ Client Connection

Responsibility Boundary

Streamlit Client

Responsible for: - Rendering UI - File selection - Basic client-side
validation - Sending HTTP requests - Displaying processing states -
Rendering streamed answers - Rendering citations/errors

The client must not contain core RAG logic.

FastAPI Server

Responsible for: - Request validation - File ingestion orchestration -
Preprocessing - Chunking - Embedding orchestration - ChromaDB access -
Retrieval - Context validation - LLM generation - Citation
construction - Output validation - Error normalization -
Logging/telemetry hooks - Security and rate limiting

Request Lifecycle

Streamlit
   │
   │ HTTP request
   ▼
FastAPI route
   │
   ▼
Validation
   │
   ▼
Service layer
   │
   ├── Ingestion service
   ├── Retrieval service
   ├── Generation service
   └── Evaluation/telemetry service
   │
   ▼
External providers / ChromaDB
   │
   ▼
Normalized API response
   │
   ▼
Streamlit

Keep route handlers thin. Business logic should live in services/modules
rather than inside FastAPI route functions.

4. Core Backend Flows

Ingestion

Upload
→ Validate
→ Parse
→ Clean using re
→ Chunk
→ Attach metadata
→ Embed
→ Persist in ChromaDB

Every chunk must retain enough metadata to reconstruct its source
citation.

Minimum metadata: - document_id - document_name - source_type -
chunk_id - page_number when available - section when available -
source_location

The exact metadata schema should remain consistent between ingestion,
retrieval, citation generation, and evaluation.

Retrieval

V1 uses:

query → embedding → ChromaDB similarity search → top 3 chunks → validation

Do not change k=3 without an experiment.

Context validation should determine whether the retrieved material is
sufficient to answer the question. If not, the generation step should
not fabricate an answer.

Generation

The generation prompt should: - Provide retrieved context explicitly. -
Require answers to remain grounded in that context. - Require citations
for supported claims. - Instruct the model to refuse when evidence is
insufficient.

The backend validates the generated output before returning it.

5. Error, Loading & Edge States

Every page must have explicit states:

Loading

Uploading

Processing

Retrieving

Generating

Evaluating

Success

Document ready

Answer generated

Evaluation completed

User Errors

Unsupported file type

Empty document

Invalid query

Document not ready

Missing required input

Server / Provider Errors

Embedding failure

Vector database failure

LLM failure

Timeout

Rate limit

Unexpected backend error

The frontend should display a useful user-facing message while avoiding
internal stack traces or secrets.

The backend should return normalized errors according to API.md.

6. Performance & Production Considerations

V1 should remain simple. Optimizations are introduced only after the
baseline is measured.

Ingestion

Embedding caching to reduce repeated API calls.

Async/parallel processing where useful.

Concurrency must respect provider, CPU, memory, and database limits.

Retrieval

Metadata filtering when it improves retrieval quality or latency.

Generation

Streaming to improve perceived latency.

Query caching for repeated queries.

Output/citation validation.

Infrastructure

Introduce only when justified: - API caching - Memoization - Lazy
loading - Code splitting - Rate limiting - API security - Robust error
handling

For each optimization record:

Baseline → Bottleneck → Change → Measurement → Trade-off → Decision

7. Design & UX Principles

The interface should feel like a serious engineering/product application
rather than a generic chatbot.

Priorities: 1. Clear document state 2. Clear answer/source relationship
3. Visible citations 4. Obvious refusal state 5. Minimal unnecessary
navigation 6. Responsive feedback during long-running operations 7.
Consistent loading/error/success states

Do not hide retrieval or citation information behind unnecessary
interactions. The user should be able to understand why an answer was
produced.

8. V1 Completion Criteria

V1 is complete only when:

PDF/DOCX/TXT ingestion works.

Cleaning and simple chunking work.

Chunks contain consistent metadata.

Embeddings are stored in ChromaDB.

Queries retrieve top 3 chunks.

Retrieved context is validated.

Answers are grounded in retrieved context.

Chunk-level and multi-source citations work.

Citation-aware refusal works.

RAGAS evaluation can measure faithfulness, context recall, and
context precision.

Latency/token/failure metrics can be captured.

Major engineering decisions are documented.

No performance improvement is claimed without measured evidence.

The frontend communicates with the backend through the API contracts
defined in API.md.

V2 and V3 are intentionally not implemented in this flow until V1
results identify the next engineering problems to solve.

Source of Truth

This document defines the application flow and client/server
responsibilities.

API.md defines API contracts.

The RAG project specification defines RAG behavior, evaluation
requirements, optimization strategy, and engineering decision
principles.

If a requirement is not specified in these sources, do not invent it.
State that the information is unknown and request clarification.