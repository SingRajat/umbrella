# Project Tracker

## Status

* 🔴 **To Do**
* 🟡 **In Progress**
* 🟢 **Completed**

---

## V1 — RAG Foundation

| Task                          | Status        |
| ----------------------------- | ------------- |
| Project/repository setup      | 🟢 Completed  |
| Backend FastAPI setup         | 🟢 Completed  |
| Frontend Streamlit setup      | 🟢 Completed  |
| API structure from `API.md`   | 🟢 Completed  |
| Landing page                  | 🟢 Completed  |
| RAG workspace UI              | 🟢 Completed  |
| Document upload               | 🟢 Completed  |
| PDF ingestion                 | 🟢 Completed  |
| DOCX ingestion                | 🟢 Completed  |
| TXT ingestion                 | 🟢 Completed  |
| Text preprocessing & cleaning | 🟢 Completed  |
| Simple chunking               | 🟢 Completed  |
| Chunk metadata                | 🟢 Completed  |
| Embedding generation          | 🟢 Completed  |
| ChromaDB integration          | 🟢 Completed  |
| Document ingestion pipeline   | 🟢 Completed  |
| Query pipeline                | 🟢 Completed  |
| Top-k=3 retrieval             | 🟢 Completed  |
| Retrieved-context validation  | 🟢 Completed  |
| LLM generation                | 🟢 Completed  |
| Chunk-level citations         | 🟢 Completed  |
| Multi-source citations        | 🟢 Completed  |
| Citation-aware refusal        | 🟢 Completed  |
| Answer/output validation      | 🟢 Completed  |
| Streaming responses           | 🟢 Completed  |
| Error handling                | 🟢 Completed  |
| Rate limiting                 | 🟢 Completed  |
| API security                  | 🟢 Completed  |

---

## V1 — Evaluation

| Task                          | Status        |
| ----------------------------- | ------------- |
| Obtain evaluation dataset     | 🔴 To Do      |
| Create evaluation pipeline    | 🟢 Completed  |
| RAGAS setup                   | 🟢 Completed  |
| Faithfulness evaluation       | 🔴 To Do      |
| Context precision evaluation  | 🔴 To Do      |
| Context recall evaluation     | 🔴 To Do      |
| Latency tracking              | 🔴 To Do      |
| Token usage tracking          | 🔴 To Do      |
| Failure/refusal tracking      | 🔴 To Do      |
| Establish baseline results    | 🔴 To Do      |
| Identify first bottleneck     | 🔴 To Do      |
| Document engineering decision | 🔴 To Do      |

---

## V1 — Optimization

| Task                             | Status   |
| -------------------------------- | -------- |
| Embedding caching                | 🔴 To Do |
| Async/parallel ingestion         | 🔴 To Do |
| Metadata filtering               | 🔴 To Do |
| Query caching                    | 🔴 To Do |
| API caching                      | 🔴 To Do |
| Memoization                      | 🔴 To Do |
| Lazy loading                     | 🔴 To Do |
| Code splitting                   | 🔴 To Do |
| Performance benchmarking         | 🔴 To Do |
| Before/after evaluation          | 🔴 To Do |
| Document optimization trade-offs | 🔴 To Do |

---

## V1 — Finalization

| Task                         | Status        |
| ---------------------------- | ------------- |
| End-to-end testing           | 🟢 Completed  |
| Edge-case testing            | 🟢 Completed  |
| API validation               | 🟢 Completed  |
| UI/UX polish                 | 🟢 Completed  |
| Production error handling    | 🟢 Completed  |
| Final evaluation             | 🔴 To Do      |
| Final performance comparison | 🔴 To Do      |
| Engineering decision log     | 🟢 Completed  |
| Resume-ready metrics         | 🔴 To Do      |
| V1 complete                  | 🟡 In Progress |

---

## Rules for Antigravity

1. Work on **one logical task at a time**.
2. Change status to **🟡 In Progress** before starting a task.
3. Change status to **🟢 Completed** only after the task is implemented and verified.
4. If blocked, keep the task **🟡 In Progress** and document the blocker.
5. Do not mark a task completed based only on code generation; verify that it works.
6. Do not start V2/V3 until V1 is completed and evaluated.
7. Update this file whenever task status changes.
8. Do not remove completed tasks; preserve the project history.
9. For optimization tasks, record measurable before/after results where applicable.
10. Follow `API.md` and `AppFlow.md` as the architecture/API source of truth.
