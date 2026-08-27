"""Citation rendering UI component."""
from typing import Any, Dict, List
import streamlit as st


def render_citations(citations: List[Dict[str, Any]]):
    """Render structured source citations in an expandable container."""
    if not citations:
        return

    with st.expander(f"📚 Sources & Citations ({len(citations)} chunks referenced)", expanded=False):
        for idx, cite in enumerate(citations, 1):
            doc_name = cite.get("doc_name", "Unknown Document")
            page_num = cite.get("page_number")
            page_info = f"p. {page_num}" if page_num is not None else "Full Document"
            sec_heading = cite.get("section_heading")
            heading_info = f" | Section: *{sec_heading}*" if sec_heading else ""
            chunk_id = cite.get("chunk_id", "N/A")
            excerpt = cite.get("text_excerpt", "").strip()

            st.markdown(f"**[{idx}] `{doc_name}`** ({page_info}{heading_info}) — *Chunk ID: `{chunk_id[:8]}...`*")
            if excerpt:
                st.caption(f'"{excerpt}"')
            st.divider()
