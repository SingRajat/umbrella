import httpx
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000/api/v1"


def render_upload_widget():
    """Renders document uploader and document list management."""
    st.subheader("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        help="Max file size 25MB. Deduplication enabled.",
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            with st.spinner("Processing, cleaning, chunking, and embedding..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    response = httpx.post(
                        f"{API_BASE_URL}/documents",
                        files=files,
                        timeout=60.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"Successfully ingested '{data['filename']}' ({data['chunk_count']} chunks)."
                        )
                        st.rerun()
                    elif response.status_code == 409:
                        st.warning("This document has already been ingested (duplicate content detected).")
                    elif response.status_code == 413:
                        st.error("File exceeds 25MB size limit.")
                    else:
                        detail = response.json().get("message", response.text)
                        st.error(f"Ingestion failed: {detail}")
                except Exception as e:
                    st.error(f"Cannot connect to backend: {e}")

    # List Ingested Documents
    st.markdown("---")
    st.markdown("#### 📑 Ingested Documents")
    try:
        res = httpx.get(f"{API_BASE_URL}/documents?page=1&page_size=20", timeout=5.0)
        if res.status_code == 200:
            docs_data = res.json()
            docs = docs_data.get("documents", [])
            if not docs:
                st.caption("No documents ingested yet.")
            else:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{doc['filename']}** ({doc['chunk_count']} chunks)")
                        st.caption(f"ID: `{doc['doc_id'][:8]}...` | Type: {doc['source_type'].upper()}")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete document"):
                            del_res = httpx.delete(f"{API_BASE_URL}/documents/{doc['doc_id']}")
                            if del_res.status_code == 200:
                                st.toast(f"Deleted {doc['filename']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
        else:
            st.caption("Unable to fetch documents from backend.")
    except Exception:
        st.caption("Backend offline or unreachable.")
