"""Upload and document management UI component."""
import requests
import streamlit as st


def render_upload_section(backend_url: str):
    """Render document upload widget and document metadata list."""
    st.subheader("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        help="Max file size: 25 MB",
        key="doc_uploader",
    )

    if uploaded_file is not None:
        if st.button("📥 Ingest Document", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{backend_url}/api/v1/documents", files=files, timeout=30)
                    if response.status_code in (200, 201):
                        data = response.json()
                        st.session_state.current_doc = data
                        st.success(f"Ingested '{data['filename']}' ({data.get('chunk_count', 0)} chunks)!")
                        st.rerun()
                    else:
                        err = response.json()
                        st.error(f"Upload failed: {err.get('message', 'Unknown error')}")
                except Exception as exc:
                    st.error(f"Connection error: {exc}")

    # Display active document state
    if st.session_state.get("current_doc"):
        doc = st.session_state.current_doc
        st.markdown("---")
        st.markdown("**Active Document:**")
        st.info(
            f"**File:** `{doc.get('filename')}`\n\n"
            f"**Doc ID:** `{doc.get('doc_id')}`\n\n"
            f"**Chunks:** `{doc.get('chunk_count', 0)}`"
        )
        if st.button("🔄 Clear Active Document", use_container_width=True):
            st.session_state.current_doc = None
            st.rerun()
