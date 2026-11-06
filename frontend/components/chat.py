"""Chat interface and message handling UI component."""
import sys
from pathlib import Path

# Ensure project root is on sys.path 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional
import requests
import streamlit as st
from frontend.components.citations import render_citations


def render_chat_section(backend_url: str, active_doc_id: Optional[str] = None):
    """Render chat conversation history and question input."""
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False

    # Header with Prominent New Chat Button
    col_header, col_btn = st.columns([4, 1])
    with col_header:
        st.subheader("Ask Your Documents")
    with col_btn:
        if st.button("New Chat", use_container_width=True, help="Clear conversation and start fresh"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_summary = None
            st.rerun()

    # Render previous conversation turns from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Determine input availability
    input_disabled = st.session_state.is_generating or not bool(active_doc_id)
    placeholder = (
        "Ask a question about your uploaded documents..."
        if active_doc_id
        else "Upload or select a document above to begin asking questions..."
    )

    # User input box
    if prompt := st.chat_input(placeholder, disabled=input_disabled):
        if st.session_state.is_generating:
            st.warning("A query is already in progress. Please wait and resend your question once it finishes.")
            return

        if not active_doc_id:
            st.warning("Please select or upload a document before asking questions.")
            return

        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send query to FastAPI backend under generation lock
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence & generating answer..."):
                st.session_state.is_generating = True
                try:
                    payload = {
                        "query": prompt,
                        "doc_id": active_doc_id,
                        "stream": False,
                        "conversation_history": st.session_state.chat_history if st.session_state.chat_history else None,
                        "conversation_summary": st.session_state.conversation_summary,
                    }
                    response = requests.post(f"{backend_url}/api/v1/query", json=payload, timeout=45)

                    if response.status_code == 200:
                        data = response.json()

                        # Sync model memory directly from backend response
                        st.session_state.chat_history = data.get("active_history", [])
                        st.session_state.conversation_summary = data.get("conversation_summary")

                        if data.get("status") == "refused":
                            answer_text = f"**Refusal:** {data.get('reason', 'Context is insufficient.')}"
                            citations = []
                        else:
                            answer_text = data.get("answer", "No answer returned.")
                            citations = data.get("citations", [])

                        st.markdown(answer_text)
                        if citations:
                            render_citations(citations)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer_text, "citations": citations}
                        )

                    else:
                        err = response.json()
                        err_msg = f"❌ **Error:** {err.get('message', 'Failed to generate answer.')}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

                except Exception as exc:
                    err_msg = f"❌ **Connection Error:** {exc}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                finally:
                    st.session_state.is_generating = False
                    st.rerun()
