"""Workspace page for Umbrella RAG application."""
import streamlit as st
from frontend.app import get_backend_url
from frontend.components.chat import render_chat_section
from frontend.components.upload import render_upload_section


def render_workspace_page():
    st.set_page_config(
        page_title="Workspace — Umbrella RAG",
        page_icon="☂️",
        layout="wide",
    )

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None

    backend_url = get_backend_url()

    st.title("📁 RAG Workspace")
    st.caption("Upload a document on the left, then ask questions on the right to receive cited answers.")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        render_upload_section(backend_url)

    with col_right:
        active_doc_id = (
            st.session_state.current_doc.get("doc_id")
            if st.session_state.get("current_doc")
            else None
        )
        render_chat_section(backend_url, active_doc_id=active_doc_id)


if __name__ == "__main__":
    render_workspace_page()
