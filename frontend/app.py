import streamlit as st

st.set_page_config(
    page_title="Umbrella RAG",
    page_icon="☂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("☂️ Umbrella RAG System")
st.markdown(
    """
Welcome to **Umbrella** — a production-grade, evidence-grounded Retrieval-Augmented Generation (RAG) system.

Use the sidebar navigation to explore:
- **1_Landing**: Product introduction and architectural pipeline overview.
- **2_Workspace**: Upload documents, browse ingested sources, and ask grounded questions with chunk-level citations.
- **3_Evaluation**: View RAGAS evaluation metrics, benchmark results, and engineering decision logs.
"""
)

# Backend status check widget
import httpx
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 System Health")
try:
    health_res = httpx.get("http://127.0.0.1:8000/api/v1/health", timeout=3.0)
    if health_res.status_code == 200:
        data = health_res.json()
        st.sidebar.success(f"Backend: **{data['status'].upper()}**")
        st.sidebar.caption(f"ChromaDB: `{data['chromadb']}` | v{data['version']}")
    else:
        st.sidebar.warning("Backend returned non-200")
except Exception:
    st.sidebar.error("Backend Offline (127.0.0.1:8000)")
