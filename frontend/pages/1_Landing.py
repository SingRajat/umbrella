import streamlit as st

st.set_page_config(page_title="Umbrella — Overview", page_icon="☂️", layout="wide")

# Hero Section
st.title("☂️ Umbrella — Evidence-Backed RAG")
st.subheader("Transform unstructured documents into verified, citation-backed answers with zero hallucination.")

st.markdown(
    """
Umbrella is a research-backed, production-oriented Retrieval-Augmented Generation system designed
to provide **provably grounded answers** with machine-verifiable chunk citations and explicit refusal contracts.
"""
)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Open RAG Workspace", type="primary", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")
with col2:
    if st.button("📊 View Engineering Metrics", use_container_width=True):
        st.switch_page("pages/3_Evaluation.py")

st.markdown("---")

# How It Works Section
st.header("⚙️ How It Works (V1 Pipeline)")
st.code(
    """
Upload (PDF/DOCX/TXT/MD)
  │
  ▼
Clean (Regex-based dehyphenation, header/footer stripping, control char removal)
  │
  ▼
Chunk (RecursiveCharacterTextSplitter with page boundary awareness)
  │
  ▼
Embed & Index (ChromaDB with built-in embeddings & SHA-256 deduplication)
  │
  ▼
Retrieve (Cosine similarity search top-k=3)
  │
  ▼
Validate Context (Relevance threshold gate — skips LLM if low relevance)
  │
  ▼
Generate (Groq API llama-3.3-70b with structured JSON schema)
  │
  ▼
Validate Citations (Machine checks chunk existence & coverage; triggers refusal if unbacked)
  │
  ▼
Grounded Answer + Verifiable Citations
""",
    language="text",
)

st.markdown("---")

# Key Architectural Capabilities
st.header("🛡️ Key Architectural Principles")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 📄 Multi-Format Ingestion")
    st.write(
        "Robust, deterministic loaders supporting PDF, DOCX, TXT, and Markdown files with SHA-256 idempotency."
    )

with c2:
    st.markdown("### 📌 Verifiable Citations")
    st.write(
        "Every claim in the generated answer directly links to specific chunk IDs, source document names, and page numbers."
    )

with c3:
    st.markdown("### 🚫 Hallucination Defense")
    st.write(
        "Dual-stage validation gates: context relevance check before LLM call + citation-existence check after generation."
    )
