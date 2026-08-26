# Project Tracker

## Status

* 🔴 **To Do**
* 🟡 **In Progress**
* 🟢 **Completed**

---

## V1 — RAG Foundation

| Task                          | Status   |
| ----------------------------- | -------- |
| Project/repository setup      | 🟢 Completed |
| Backend FastAPI setup         | 🔴 To Do |
| Frontend Streamlit setup      | 🔴 To Do |
| API structure from `API.md`   | 🔴 To Do |
| Landing page                  | 🔴 To Do |
| RAG workspace UI              | 🔴 To Do |
| Document upload               | 🔴 To Do |
| PDF ingestion                 | 🔴 To Do |
| DOCX ingestion                | 🔴 To Do |
| TXT ingestion                 | 🔴 To Do |
| Text preprocessing & cleaning | 🔴 To Do |
| Simple chunking               | 🔴 To Do |
| Chunk metadata                | 🔴 To Do |
| Embedding generation          | 🔴 To Do |
| ChromaDB integration          | 🔴 To Do |
| Document ingestion pipeline   | 🔴 To Do |
| Query pipeline                | 🔴 To Do |
| Top-k=3 retrieval             | 🔴 To Do |
| Retrieved-context validation  | 🔴 To Do |
| LLM generation                | 🔴 To Do |
| Chunk-level citations         | 🔴 To Do |
| Multi-source citations        | 🔴 To Do |
| Citation-aware refusal        | 🔴 To Do |
| Answer/output validation      | 🔴 To Do |
| Streaming responses           | 🔴 To Do |
| Error handling                | 🔴 To Do |
| Rate limiting                 | 🔴 To Do |
| API security                  | 🔴 To Do |

---

## V1 — Evaluation

| Task                          | Status   |
| ----------------------------- | -------- |
| Obtain evaluation dataset     | 🔴 To Do |
| Create evaluation pipeline    | 🔴 To Do |
| RAGAS setup                   | 🔴 To Do |
| Faithfulness evaluation       | 🔴 To Do |
| Context precision evaluation  | 🔴 To Do |
| Context recall evaluation     | 🔴 To Do |
| Latency tracking              | 🔴 To Do |
| Token usage tracking          | 🔴 To Do |
| Failure/refusal tracking      | 🔴 To Do |
| Establish baseline results    | 🔴 To Do |
| Identify first bottleneck     | 🔴 To Do |
| Document engineering decision | 🔴 To Do |

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

| Task                         | Status   |
| ---------------------------- | -------- |
| End-to-end testing           | 🔴 To Do |
| Edge-case testing            | 🔴 To Do |
| API validation               | 🔴 To Do |
| UI/UX polish                 | 🔴 To Do |
| Production error handling    | 🔴 To Do |
| Final evaluation             | 🔴 To Do |
| Final performance comparison | 🔴 To Do |
| Engineering decision log     | 🔴 To Do |
| Resume-ready metrics         | 🔴 To Do |
| V1 complete                  | 🔴 To Do |

---

## Rules for Antigravity

1. Work on **one logical task at a time**.

2. Implement **only ONE 🔴 To Do task per execution cycle**.

3. Do not implement, scaffold, or modify code for future 🔴 To Do tasks.

4. Change status to **🟡 In Progress** before starting a task.

5. Before implementation, briefly explain what will be changed and why.

6. Implement the current task completely before moving to another task.

7. Run the relevant tests or verification after implementation.

8. Change status to **🟢 Completed** only after the task is implemented and verified.

9. If blocked, keep the task **🟡 In Progress** and document the blocker.

10. Do not mark a task completed based only on code generation; verify that it works.

11. After completing a task, **STOP and wait for the next instruction**. Do not automatically start the next To Do task.

12. Do not batch multiple tracker tasks into one implementation.

13. Do not create placeholder implementations for future tasks just to mark them as completed.

22. For optimization tasks, record measurable **before/after results** where applicable.

23. Use the same **50–60 ground-truth questions** when comparing RAG strategies.

24. Test major RAG optimization changes **one at a time** before combining them.

26. Do not start V2/V3 until V1 is completed and evaluated.

27. Update this file whenever task status changes.

28. Do not remove completed tasks; preserve the project history.

29. Follow `API.md`, `AppFlow.md`, and `Guardrails.md` as the architecture, API, and project constraint source of truth.
