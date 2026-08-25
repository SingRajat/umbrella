import httpx
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="Umbrella — Evaluation Dashboard", page_icon="☂️", layout="wide")

st.title("📊 Evaluation & Engineering Dashboard")
st.markdown(
    """
This dashboard displays **reproducible, measurable RAGAS metrics** and tracks historical experiment artifacts.
No metric is fabricated or assumed — only recorded evaluation runs against ground-truth datasets appear here.
"""
)

# Trigger Evaluation Run Section
with st.expander("⚡ Run New Evaluation Benchmark", expanded=False):
    st.markdown("Trigger an offline RAGAS evaluation run against `eval/datasets/`.")
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Executing RAGAS benchmark..."):
            try:
                res = httpx.post(f"{API_BASE_URL}/eval/run", json={}, timeout=120.0)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Evaluation run completed! Run ID: `{data['run_id']}` (Config: `{data['config_hash']}`)")
                    st.rerun()
                else:
                    st.error(f"Evaluation run failed: {res.text}")
            except Exception as e:
                st.error(f"Cannot connect to backend: {e}")

st.markdown("---")

# Metrics Summary Table
st.header("📈 Experiment Comparison Matrix")

# Fetch historical runs
from src.eval.metrics_store import metrics_store
runs = metrics_store.list_runs()

if not runs:
    st.info("ℹ️ No historical evaluation runs recorded yet. Place your 50–60 ground-truth questions in `eval/datasets/` and run a benchmark.")
    # Show comparison schema table as skeleton
    st.markdown("#### Baseline vs Experiment Target Schema")
    st.table(
        [
            {"Metric": "Faithfulness", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
            {"Metric": "Context Precision", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
            {"Metric": "Context Recall", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
            {"Metric": "Answer Relevancy", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
            {"Metric": "Retrieval Latency (p50)", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
            {"Metric": "End-to-End Latency", "Baseline (V1)": "Pending", "Experiment (V2)": "TBD", "Delta": "-"},
        ]
    )
else:
    table_data = []
    for r in runs:
        m = r.get("metrics", {})
        table_data.append(
            {
                "Run ID": r.get("run_id"),
                "Config Hash": r.get("config_hash"),
                "Created At": r.get("created_at"),
                "Faithfulness": m.get("faithfulness", "N/A"),
                "Precision": m.get("context_precision", "N/A"),
                "Recall": m.get("context_recall", "N/A"),
                "Relevancy": m.get("answer_relevancy", "N/A"),
            }
        )
    st.dataframe(table_data, use_container_width=True)

st.markdown("---")

# Engineering Decision Log Section
st.header("📝 Engineering Decision Log")
st.markdown(
    """
Every change to the system follows the **Problem → Hypothesis → Experiment → Decision → Trade-off → Result** framework.
"""
)

with st.container():
    st.markdown("#### Decision 001: Baseline Vector Database Selection")
    st.caption("Date: 2026-08-26 | Component: Storage")
    st.write(
        "- **Problem:** Need a lightweight, self-contained, embeddable vector database for single-machine local V1 execution.\n"
        "- **Hypothesis:** ChromaDB with built-in embeddings provides zero-configuration local persistence without network latency or external infrastructure overhead.\n"
        "- **Decision:** Use ChromaDB wrapped in `VectorStoreClient` protocol.\n"
        "- **Trade-off:** Single-process SQLite metadata backend; acceptable for V1 local single-user research mode, with PostgreSQL abstraction in place for future multi-tenant scale."
    )
