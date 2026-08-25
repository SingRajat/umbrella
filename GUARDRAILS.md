# Umbrella Project Guardrails

## 1. Core Architectural Guardrails
1. **Separation of Concerns**: The Streamlit frontend MUST NOT directly interact with ChromaDB, LLM APIs, or vector databases. All business logic lives in the FastAPI backend.
2. **LCEL Composition**: Ingestion and Query pipelines MUST be composed using LangChain LCEL `Runnable` chains (`src/chains/`) while keeping stages unit-testable and modular in `src/ingestion/` and `src/query/`.
3. **Storage Abstraction**: ChromaDB integration lives in `src/storage/` so storage logic is shared cleanly across ingestion and query, allowing future PostgreSQL/hybrid expansion without rewriting pipeline stages.
4. **Idempotency & Deduplication**: Document ingestion must check file content hash (`SHA-256`) before reprocessing to prevent duplicate chunk storage.
5. **No Speculative Complexities**: Stick strictly to V1 scope (no unverified re-rankers, no multi-tenant auth, no premature optimizations without baseline measurement).

## 2. Hallucination Defense & Quality Guardrails
1. **Relevance Gating**: Queries must pass context validation; if no retrieved context meets requirements or corpus is empty, the system must trigger an explicit refusal rather than fabricating answers.
2. **Context-Only Grounding**: System prompts strictly require answers to be derived exclusively from retrieved chunks.
3. **Structured Machine-Verifiable Citations**: LLMs must return structured outputs specifying exact `chunk_id` citations.
4. **Citation Existence Validation**: Any cited `chunk_id` must match actual retrieved chunks. Hallucinated citations trigger a refusal in strict mode.
5. **No Fabricated Confidence**: Confidence scores must not be generated via arbitrary LLM self-assessment.

## 3. Engineering & Decision Principles
1. **Evidence-Based Engineering**: Any performance or architectural claim requires measured data (`Problem → Hypothesis → Experiment → Decision → Trade-off → Result`).
2. **One Variable at a Time**: In evaluation runs, alter only one parameter relative to baseline.
3. **Artifact Permanence**: Evaluation metrics and experiment results are permanent records (`eval/results/`), never treated as evictable cache.
4. **Zero Assumptions / Conflict Resolution**: If requirements in documentation conflict or are ambiguous, halt and obtain explicit user alignment before writing code.
