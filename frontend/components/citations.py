import streamlit as st
from typing import Any


def render_citations_panel(citations: list[dict[str, Any]]):
    """Renders structured citations with document details and source excerpts."""
    if not citations:
        return

    st.markdown("### 📚 Source Citations")
    for idx, cite in enumerate(citations, start=1):
        doc_name = cite.get("doc_name", "Unknown Document")
        page = cite.get("page_number")
        section = cite.get("section_heading")
        excerpt = cite.get("text_excerpt", "")
        chunk_id = cite.get("chunk_id", "")

        header_tags = [f"**[{idx}] {doc_name}**"]
        if page:
            header_tags.append(f"Page {page}")
        if section:
            header_tags.append(f"Section: *{section}*")

        with st.expander(" | ".join(header_tags), expanded=False):
            st.markdown(f"**Chunk ID:** `{chunk_id}`")
            st.info(f"\"{excerpt}\"")
