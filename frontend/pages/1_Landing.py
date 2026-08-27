"""Landing page for Umbrella RAG application."""
import streamlit as st


def render_landing_page():
    st.set_page_config(
        page_title="Umbrella — Citation-Aware RAG",
        page_icon="☂️",
        layout="wide",
    )

    # --- Hero Section ---
    st.title("☂️ Umbrella RAG")
    st.subheader("Production-oriented, evidence-backed question answering over your documents.")
    st.markdown(
        """
        Umbrella is an open-source RAG system designed for **measurable evaluation, hallucination defense, and verifiable citations**. 
        Ask questions across your technical documents, contracts, and papers and get answers grounded strictly in factual retrieved context.
        """
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Go to Workspace", type="primary", use_container_width=True):
            st.switch_page("pages/2_Workspace.py")
    with col2:
        st.caption("👈 Or click on '2_Workspace' in the sidebar navigation")

    st.markdown("---")

    # --- How It Works Pipeline ---
    st.header("⚙️ How It Works (V0 Baseline Pipeline)")
    st.markdown(
        """
        ```
        [Document Upload] ──► [Regex Cleaning] ──► [Recursive Chunking] ──► [ChromaDB Indexing]
                                                                                     │
        [Verifiable Answer] ◄── [Groq LLM Generation] ◄── [Top-K=3 Retrieval] ◄─────┘
        ```
        """
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown("**1. Ingestion**")
        st.write("Parses PDF, DOCX, TXT, and MD files with structural cleaning.")
    with col_b:
        st.markdown("**2. Chunking & Indexing**")
        st.write("Splits content with character overlap and persists vectors in ChromaDB.")
    with col_c:
        st.markdown("**3. Precision Retrieval**")
        st.write("Extracts top-k most relevant evidence chunks for each user query.")
    with col_d:
        st.markdown("**4. Grounded Generation**")
        st.write("Generates answers with exact chunk-level citations, or refuses if evidence is lacking.")

    st.markdown("---")

    # --- Key Capabilities & Engineering Principles ---
    st.header("🎯 Key Capabilities & Design Principles")

    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown("#### 🛡️ Hallucination Defense")
        st.markdown(
            """
            - **Strict Context Grounding:** Prompts instruct the LLM to use *only* retrieved evidence.
            - **Machine-Verifiable Citations:** Every claim links directly to its source chunk.
            - **Citation-Aware Refusal:** Explicitly informs the user if context is insufficient instead of guessing.
            """
        )

        st.markdown("#### 📂 Multi-Format Support")
        st.markdown(
            """
            - **PDF:** Native extraction via `pdfplumber` / `pypdf`.
            - **DOCX & TXT / MD:** Preserves headings, lists, and metadata structures.
            """
        )

    with cap_col2:
        st.markdown("#### 📊 Measurable Evaluation")
        st.markdown(
            """
            - **RAGAS Benchmark Harness:** Measures faithfulness, answer correctness, and context precision.
            - **Hypothesis-Driven Engineering:** Every optimization (semantic chunking, reranking, hybrid search) is benchmarked against frozen ground truth.
            """
        )

        st.markdown("#### ⚡ Fast & Production-Ready")
        st.markdown(
            """
            - **High-Speed Inference:** Powered by Groq API (`llama-3.3-70b-versatile`).
            - **Modular LCEL Architecture:** Decoupled ingestion, storage, retrieval, and generation chains.
            """
        )


if __name__ == "__main__":
    render_landing_page()
