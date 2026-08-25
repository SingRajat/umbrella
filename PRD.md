# PRD: Production-Grade RAG System (V1) — Resume Project

## 1. Overview & Goal

Build a modular, production-oriented Retrieval-Augmented Generation (RAG) system whose primary artifact is not "a chatbot that works" but **evidence of engineering judgment**: baseline → measured evaluation → identified bottleneck → hypothesis-driven change → re-measured result → documented trade-off.

V1 scope only. V2/V3 are explicitly out of scope until V1 is built and evaluated against real numbers. No feature is implemented "because it's best practice" — every feature must map to an identified problem or a measurable benefit.

**Primary success criterion:** every non-trivial technical decision in the repo is traceable to a `Problem → Hypothesis → Experiment → Decision → Trade-off → Result` entry, backed by RAGAS metrics. (The decision-log *format and content* is tracked as a separate artifact, outside this PRD's scope — see §14.)

**Secondary success criterion:** the project reads well as a resume artifact — clear before/after numbers, clear rationale, clear trade-offs, nothing implemented without justification.

---

## 2. System Architecture

Two independent, composable LCEL pipelines, connected only through the vector store. This separation is deliberate: it lets ingestion and query be developed, tested, and evaluated independently, and it lets any single stage be swapped without touching the rest of the chain.

```
[Ingestion Pipeline]
 Upload (PDF/DOCX/TXT)
   → Loader (format-specific)
   → Cleaner (regex-based, `re`)
   → Chunker (simple: fixed-size + overlap)
   → Metadata Tagger
   → Embedder (with cache)
   → Vector Store Writer

[Query Pipeline]
 User Query
   → Retriever (k=3, similarity search)
   → Context Validator (relevance/threshold check)
   → Prompt Builder (context + citation tags)
   → LLM Generation (structured output)
   → Answer Validator (citation-support check)
   → Response (answer + citations, or refusal)
```

Both pipelines are built as LCEL `Runnable` chains (`RunnableLambda`, `RunnableSequence`, `RunnableParallel` where applicable) so that:
- Any stage can be unit-tested in isolation with a fixed input/output contract.
- Any stage can be swapped for an alternative implementation and A/B evaluated without touching upstream/downstream code.
- The chain itself is declarative and easy to reason about in a design review — a strong resume signal.

### 2.1 Repo Structure

```
/src
  /ingestion
    loaders.py        # PDF/DOCX/TXT → RawDocument
    cleaner.py         # regex cleaning, pure functions
    chunker.py          # fixed-size + overlap chunker
    metadata.py         # chunk metadata construction
    embedder.py           # embedding interface + cache wrapper
    vectorstore.py         # VectorStoreClient interface + implementation
  /query
    retriever.py         # similarity search, k param
    context_validator.py  # relevance threshold gate
    prompt.py              # prompt template + assembly
    generator.py             # LLM call, structured output parsing
    output_validator.py       # citation-support check, refusal logic
  /chains
    ingestion_chain.py     # LCEL composition only, no business logic
    query_chain.py           # LCEL composition only, no business logic
  /eval
    ragas_runner.py         # runs RAGAS against eval set
    datasets.py               # eval set loader (dataset TBD, see §13)
    metrics_store.py           # persists run results with config hash
  /api
    routes.py                 # FastAPI routes
    schemas.py                  # Pydantic request/response models
    middleware.py                 # rate limiting, auth, correlation IDs
  /config
    settings.py                   # pydantic-settings, env-driven
  /common
    logging.py                      # structured JSON logging setup
    errors.py                         # typed exception hierarchy
    cache.py                            # generic cache interface (embedding + query cache reuse this)
/tests
  /unit          # cleaner, chunker, validators — pure function tests
  /integration   # end-to-end ingestion_chain, end-to-end query_chain
/eval
  /results       # {run_id}.json, one per RAGAS run
  /datasets      # eval set once provided (see §13)
/decision_log     # out of scope for this PRD, tracked separately
```

### 2.2 Design Principle: Interfaces Before Implementations

Every swappable component (`Embedder`, `VectorStoreClient`, `Generator`, `Chunker`) is defined as an abstract interface first, with a single concrete implementation for V1. This is what makes the "Option A vs Option B" evaluation pattern possible later without a rewrite — a component can be re-implemented and benchmarked against the same test harness.

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class VectorStoreClient(Protocol):
    def upsert(self, chunks: list[ChunkRecord]) -> None: ...
    def query(self, vector: list[float], k: int, filters: dict | None) -> list[RetrievedChunk]: ...
    def delete_by_doc_id(self, doc_id: str) -> None: ...

class Generator(Protocol):
    def generate(self, prompt: str) -> AnswerWithCitations: ...
```

---

## 3. Ingestion Pipeline — Implementation Detail

### 3.1 Document Loading

- **Formats:** PDF (`pypdf` or `pdfplumber` — `pdfplumber` preferred if layout/heading detection is needed for `section_heading` metadata), DOCX (`python-docx`), TXT (native read).
- Loader interface: `load(file_path: str, source_type: str) -> RawDocument`.
- `RawDocument` shape:
```json
{
  "doc_id": "uuid4",
  "filename": "string",
  "source_type": "pdf|docx|txt",
  "uploaded_at": "ISO8601",
  "pages": [{"page_num": 1, "text": "raw extracted text"}]
}
```
- `doc_id` is a UUID4 minted at upload time and persists for the document's lifetime (used for deletion, filtering, and citation linkage).
- TXT files have no native page concept — treat the whole file as `page_num: null` (a single "page").
- DOCX files similarly lack hard page breaks — `page_num: null`; section detection instead relies on heading styles (see §3.2).
- Failure modes to handle explicitly: unreadable/corrupted file → typed `IngestionError`, surfaced to the API layer as a 4xx with a clear message; empty extracted text → same treatment (do not silently create a doc with zero chunks).

### 3.2 Cleaning (`re`-based)

Deterministic, testable regex cleaning applied per page/section **before** chunking. Implemented as pure functions so they're independently unit-testable and independently benchmarkable if cleaning quality is later suspected as a bottleneck.

Cleaning steps:
1. **Header/footer stripping** — detect lines that repeat identically (or near-identically) across ≥3 pages at the same relative position; strip them. This catches running headers, footers, and page-number artifacts without hardcoding patterns.
2. **Whitespace normalization** — collapse repeated whitespace/newlines to single spaces/newlines; strip leading/trailing whitespace per line.
3. **Control character removal** — strip non-printable characters left over from PDF extraction.
4. **De-hyphenation** — rejoin line-broken words (`word-\nwrap` → `wordwrap`) via regex on `-\n` boundaries, with a guard against legitimately hyphenated compound words (heuristic: only rejoin if the resulting joined string is a more "word-like" token than the split form — simple length/alpha check, not a dictionary lookup for V1).
5. **Structure preservation** — headings (detected via DOCX paragraph style, or PDF font-size heuristics if available) and bullet/numbered-list markers are **not** stripped; they are preserved and passed through to the metadata tagger (§3.4) as `section_heading` candidates.

`clean(text: str) -> str` is a pure function with no I/O — this is a deliberate testability requirement.

### 3.3 Chunking (Baseline — Simple)

- **Strategy:** fixed-size chunking with overlap, computed on token count (not raw characters) using `tiktoken` for consistent sizing across documents regardless of language density.
- **Baseline parameters (assumption, to be logged as a decision once benchmarked):** `chunk_size = 800 tokens`, `chunk_overlap = 100 tokens`.
- Chunking respects page boundaries where possible — a chunk does not silently span two documents, but it *may* span two pages of the same document (page boundary is metadata, not a hard split point, since sentences/paragraphs routinely cross page breaks in real documents).
- Interface: `chunk(cleaned_pages: list[CleanedPage]) -> list[Chunk]`, where each `Chunk` retains a reference to which page(s) it was drawn from (needed for `page_number` metadata, which may be a single page or a range).
- **Explicitly deferred to V2:** semantic chunking, recursive character splitting, sentence-boundary-aware chunking, table-aware chunking. These are only justified if V1 evaluation shows context-recall/precision problems traceable to chunk boundaries — implementing them upfront would violate the "problem before solution" rule.

### 3.4 Metadata Schema (per chunk)

Every chunk is tagged with metadata at ingestion time and stored **together with the embedding** in the vector DB payload (not in a separate lookup table, to avoid a join on the query hot path in V1).

```json
{
  "chunk_id": "uuid4",
  "doc_id": "uuid4",
  "doc_name": "string",
  "source_type": "pdf|docx|txt",
  "page_number": "int | [int,int] | null",
  "section_heading": "string | null",
  "chunk_index": "int",
  "char_start": "int",
  "char_end": "int",
  "domain_tag": "string | null",
  "ingested_at": "ISO8601 timestamp"
}
```

- `chunk_index` is the sequential position of the chunk within the document — used for ordering and for reconstructing surrounding context if ever needed for debugging.
- `char_start`/`char_end` are offsets into the *cleaned* document text, enabling exact traceability back to source for citation rendering or highlighting.
- `domain_tag` (medical/legal/banking/technical/general/other) is **user-supplied at upload time, optional, and never used to alter core pipeline logic** — see §8.

### 3.5 Embedding & Storage

- Embedder interface: `embed(texts: list[str]) -> list[list[float]]`, batched (not one call per chunk) to control API call volume.
- **Embedding cache (justified, always-on V1 feature):** re-ingesting an already-seen chunk (e.g., re-uploading the same document, or overlapping content across documents) is a real, avoidable cost. Cache key = `hash(model_name + cleaned_chunk_text)`; cache value = embedding vector. Cache backend: local key-value store for V1 (e.g., SQLite or disk-backed dict); swappable to Redis if concurrent multi-user ingestion becomes a bottleneck (measured, not assumed).
  - **Metric to capture:** cache hit rate, API calls avoided, ingestion time delta, estimated $ saved — reported as an explicit before/after number.
- Vector store interface (`VectorStoreClient`) abstracts the concrete DB so Chroma/Qdrant/pgvector are interchangeable behind `upsert`, `query`, `delete_by_doc_id`. Concrete choice is an open question (§13), not because the interface changes, but because deployment target affects which is simplest to run for a resume demo.
- `upsert` writes vector + full metadata payload in one call; `delete_by_doc_id` must cascade-delete all chunks belonging to a document (needed for the document-deletion API in §11).

---

## 4. Query Pipeline — Implementation Detail

### 4.1 Retrieval

- `retriever.query(query: str, k: int = 3, filters: dict | None = None) -> list[RetrievedChunk]`.
- Similarity metric: cosine similarity (standard for the embedding models in scope).
- `k=3` is the **fixed baseline**. Any change to `k` is treated as a major variable change and requires its own paired before/after RAGAS run — per the "don't change multiple variables at once" rule, `k` changes must not be bundled with, e.g., a chunk-size change in the same experiment.
- `RetrievedChunk` shape: `{chunk_id, doc_id, text, metadata, similarity_score}`.

### 4.2 Context Validation

A relevance gate runs **before** any chunk reaches the LLM — this is the first layer of hallucination defense and also a cost-control mechanism (no wasted LLM call on clearly irrelevant context).

- Configurable minimum similarity threshold (assumption: start at `0.5` cosine similarity, tune based on eval results — this is exactly the kind of parameter that gets a documented before/after).
- If **zero** chunks pass the threshold → short-circuit directly to a refusal response, skipping the LLM call entirely.
- If some chunks pass and some don't → only the passing chunks are forwarded; the dropped ones are still logged (chunk_id + score) for debugging/observability, not silently discarded from the trace.

### 4.3 Prompt Construction

- System prompt contract (non-negotiable, since it's central to the hallucination-prevention requirement):
  1. Answer **only** using the provided context chunks.
  2. Every factual claim must be tagged with the `chunk_id` (or `[doc_name, p.X]`) it came from.
  3. If the context does not fully support an answer, say so explicitly rather than filling gaps from general knowledge.
- Prompt assembly is a pure `RunnableLambda`: `build_prompt(query: str, chunks: list[RetrievedChunk]) -> str`, independently unit-testable against fixed inputs (important since prompt wording is itself a tunable variable subject to future A/B testing).
- Each chunk is rendered into the prompt with its citation tag inline, e.g. `[chunk_id=abc123 | doc="policy.pdf" | p.4]: <chunk text>`.

### 4.4 Generation

- `Generator.generate(prompt: str) -> AnswerWithCitations`, structured output requested from the LLM (JSON mode / tool-call schema, not free-text parsing) to make downstream validation mechanical:
```json
{
  "answer": "string",
  "citations": ["chunk_id", "..."],
  "confidence_note": "string | null"
}
```
- Structured output is a deliberate choice over regex-parsing free text — it removes an entire class of parsing bugs and makes the output validator (§4.5) a simple set-membership check rather than a fragile string-matcher.
- LLM choice is pluggable behind the `Generator` interface; concrete provider/model is an open question (§13) tied to cost/latency constraints for a demo environment, not a core architectural decision.

### 4.5 Answer / Output Validation

Two independent checks, both required:
1. **Citation-existence check:** every `chunk_id` in `citations` must correspond to a chunk that was actually retrieved *and* passed the relevance gate. Any citation pointing to a non-existent or filtered-out chunk is treated as a hallucinated citation.
2. **Citation-coverage check (best-effort for V1):** flag (not necessarily block) answers where the `citations` list is empty but `answer` is non-trivial — this is a strong hallucination signal worth logging even if full claim-level NLI-based verification is deferred to V2.

Default behavior on validation failure: **refuse** rather than silently strip the bad citation and return a partial answer — this default is chosen because a partial, silently-edited answer is harder for a user to trust-verify than an explicit refusal. This is configurable (`strict_refusal: bool` in config) so the trade-off (safety vs. answer availability) can itself be evaluated later.

### 4.6 Refusal Contract

Refusal is a first-class response type, not an HTTP error:
```json
{
  "status": "refused",
  "reason": "insufficient_context | low_relevance | unsupported_claim",
  "retrieved_chunk_ids": ["..."],
  "query": "string"
}
```
Refusals are logged identically to successful answers (same trace ID, same metrics pipeline) so **refusal rate** is a first-class, trackable metric (§6), not an edge case swept aside.

---

## 5. Citation System Requirements

- **Chunk-level:** every claim in the final answer maps to at least one `chunk_id`; the API response resolves `chunk_id → {doc_name, page_number, section_heading}` for display, so the frontend never has to re-fetch metadata separately.
- **Multi-source citations:** a single answer may legitimately draw on multiple documents; the response groups citations by `doc_id` for readability (e.g., "Supported by: policy.pdf p.4, faq.docx p.1").
- **Citation-aware refusal:** even when the system refuses to answer, the response surfaces which chunks *were* retrieved and why they were insufficient — critical for debugging retrieval quality and for user trust (the user can see the system looked, rather than assuming it did nothing).
- Citation resolution is a pure lookup against stored chunk metadata — no re-computation, no re-embedding, kept cheap and synchronous.

---

## 6. Evaluation (RAGAS)

- **Framework:** RAGAS, run against a fixed, versioned evaluation set (dataset itself is an open input — see §13; do not assume or substitute a dataset).
- **Core metrics (mandatory on every run):**
  - Faithfulness
  - Context Recall
  - Context Precision
- **Secondary metrics (tracked where useful, not mandatory for every micro-change):**
  - Answer Relevance
  - Retrieval latency (p50/p95)
  - End-to-end latency (p50/p95)
  - Token usage / estimated cost per query
  - Refusal rate (including breakdown by refusal reason)
- **Reproducibility contract:** every run is tagged with a config hash covering `{chunk_size, chunk_overlap, k, similarity_threshold, embedding_model, llm_model, prompt_version}`. Results are persisted to `/eval/results/{run_id}.json`. No metric is ever cited in project documentation without a corresponding stored run file — this is what makes every "before → after" claim auditable rather than asserted.
- **Baseline run:** the very first thing executed once the V1 pipeline is functionally complete, *before* any optimization work begins. This run's numbers are the fixed reference point every later change is measured against. Baseline config and results are frozen/tagged (e.g., `run_id: baseline-v1`) and never silently overwritten.
- **Experiment discipline:** one variable changes per run relative to its comparison baseline. If two variables must change together, that is itself called out and justified in the run's notes — not treated as routine.

---

## 7. Hallucination Prevention (Cross-Cutting Summary)

This is not a single feature but the sum of several layers already specified above — restated here as an explicit checklist so it's verifiable as a requirement:

1. Context validation gate (§4.2) — no LLM call on irrelevant context.
2. Prompt-level constraint (§4.3) — answer only from context, cite everything.
3. Structured output (§4.4) — citations are machine-checkable, not prose to parse.
4. Citation-existence validation (§4.5) — hallucinated citations are caught, not trusted.
5. Explicit refusal contract (§4.6) — "I don't have enough evidence" is a designed response type, not a fallback error message.

Each layer is independently testable and independently toggleable in config, so their individual contribution to faithfulness score can be measured in isolation (e.g., run eval with the relevance gate disabled vs. enabled, holding everything else constant) — itself a strong candidate experiment for the project's write-up.

---

## 8. User Document Upload

- Endpoint accepts PDF/DOCX/TXT via multipart upload; runs the ingestion chain **synchronously** for V1 (simplicity is preferred while ingestion latency is untested — async job queue is a V2 candidate, contingent on measured latency being a real problem, not assumed upfront).
- Supported domains: medical, technical, legal, banking, general — but **no domain-specific logic lives in the core pipeline**. All documents flow through the identical loader → cleaner → chunker → embedder path regardless of domain.
- `domain_tag` is accepted as optional user-supplied metadata at upload time (free-text or constrained enum — assumption: constrained enum for consistency, with `"general"` as default). It is stored on every chunk from that document and is usable purely as a **retrieval filter** (§9), never as a branch condition inside cleaning/chunking/prompting logic. This constraint exists specifically so the system doesn't accumulate hidden, hard-to-test domain-specific branches.
- Upload size/type limits enforced at the API layer (assumption: 25MB per file, extensions restricted to `.pdf/.docx/.txt`), returning a clear 4xx on violation rather than a pipeline-level crash.

---

## 9. Performance Optimization Features (V1-eligible)

Each row below is implemented **only** under its stated trigger condition, and each requires a documented before/after measurement — no optimization ships on vibes.

| Feature | Layer | Trigger condition | What gets measured |
|---|---|---|---|
| Embedding cache | Ingestion | Always on — low risk, trivially measurable, near-certain win for repeated/overlapping content | API calls avoided, ingestion time delta, cost delta |
| Async ingestion (`asyncio`) | Ingestion | Only if batch-ingestion latency is *measured* as a bottleneck (e.g., multi-file upload taking noticeably long); concurrency bounded to respect embedding-provider rate limits, not maximized blindly | Throughput (docs/min), latency before vs. after, error rate under concurrency |
| Metadata filtering | Retrieval | Only after baseline eval shows precision issues attributable to cross-document noise (e.g., multi-doc corpus with topically overlapping content) | Context precision before/after, latency delta, recall trade-off (filtering can hurt recall if filters are too strict) |
| Streaming responses | Generation | Always on — pure perceived-latency UX win with no accuracy trade-off, cheap to implement via LLM streaming API + SSE/websocket | Perceived time-to-first-token (no faithfulness trade-off expected, but still logged) |
| Query caching | Generation | Only after repeated/near-duplicate query patterns are actually observed in eval or usage logs — not implemented speculatively | Cache hit rate, latency improvement, and an explicit invalidation rule: cache entries tied to a document are invalidated on that document's re-ingestion or deletion |
| Output validation | Generation | Always on — core to the hallucination-prevention requirement, not optional | Rate of caught unsupported claims, false-refusal rate (cases where validation over-triggers on a legitimately supported answer) |

Metadata filtering interface: `retriever.query(query, k, filters={"domain_tag": "legal", "doc_id": "..."})` — filters are passed straight to `VectorStoreClient.query`, so no retrieval-layer logic needs to change when filters are added, only the filter dict construction at the API boundary.

---

## 10. Production Considerations (V1 baseline, deliberately not gold-plated)

- **Config management:** `pydantic-settings` / `.env`-driven — no hardcoded model names, API keys, thresholds, or chunk parameters anywhere in pipeline code.
- **Logging:** structured JSON logs at each pipeline stage boundary (ingestion: load/clean/chunk/embed/store; query: retrieve/validate/prompt/generate/validate), each entry carrying the correlation ID.
- **Error handling:** a typed exception hierarchy (`IngestionError`, `RetrievalError`, `GenerationError`, `ValidationError`) mapped to specific HTTP status codes at the API boundary — no bare `except Exception` swallowing errors silently.
- **Graceful failure / edge cases explicitly handled:**
  - Malformed or corrupted file upload.
  - Document that extracts to empty/near-empty text.
  - Embedding provider timeout or rate-limit response.
  - Vector store timeout or connection failure.
  - Query against a corpus with zero ingested documents.
  - LLM returns malformed structured output (fallback: retry once with a stricter format reminder, then refuse if still malformed).
- **Testability:** every pure-function component (cleaner, chunker, prompt builder, validators) has unit tests with fixed input/output fixtures; each full pipeline (ingestion, query) has at least one integration test running against a small fixture document set.
- **Observability-readiness:** structured logs and per-stage timing are designed to be pluggable into a tracing backend later (e.g., OpenTelemetry) without redesign — but no tracing backend is wired up in V1 itself unless justified by an actual debugging need.

Deliberately **not** built in V1 without justification: authentication/authorization beyond basic API-key check, multi-tenant isolation, horizontal scaling/load balancing, async job queues, full observability stack. These are reasonable production features in general, but none are justified by a V1-scale resume demo — flagging them here so their absence is a documented decision, not an oversight.

---

## 11. Explicitly Out of Scope for V1

- Semantic/recursive/sentence-aware/table-aware chunking, re-ranking, hybrid (BM25 + vector) search, query rewriting/expansion, multi-hop retrieval, agentic tool use, conversational memory across turns.
- Async ingestion job queue, multi-tenant auth, horizontal scaling.
- Full engineering decision-log documentation — tracked separately, not part of this PRD's current scope per instruction.

All of the above are **candidates** for V2, contingent entirely on what the V1 baseline evaluation actually reveals as the bottleneck — none are pre-committed.

---

## 12. Open Questions / Blocking Inputs

1. **Evaluation dataset / document repository** — required before any RAGAS run can execute, and explicitly *not* to be assumed or substituted. Must be supplied by the project owner before evaluation work begins.
2. **Vector store choice** (Chroma vs. Qdrant vs. pgvector) — functionally interchangeable behind `VectorStoreClient`; the concrete pick should follow from deployment target (local demo vs. hosted), not from a technical constraint.
3. **Embedding + LLM provider** — left pluggable by design; concrete choice driven by cost/latency budget for the demo environment, decided once the baseline is being built rather than upfront.
4. **`API.md`** — not present in the referenced material; §10's API design is a provisional inference and should be reconciled once the real file is available.

---

## 13. Notes on Scope Exclusions (per current instructions)

The **Engineering Decision Log** (per-component Option A vs. Option B comparisons: embedding model, vector DB, chunk size/overlap, `k`, retrieval strategy, LLM, LCEL, caching, async processing, validation strategy) is a real and required part of the overall project, but is **intentionally excluded from this PRD's content** and tracked as a separate, ongoing artifact (`/decision_log`) populated as each decision is actually made and measured — not pre-written speculatively here.