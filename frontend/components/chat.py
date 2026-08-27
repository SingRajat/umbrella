"""Chat interface and message handling UI component."""
from typing import Optional
import requests
import streamlit as st
from frontend.components.citations import render_citations


def render_chat_section(backend_url: str, active_doc_id: Optional[str] = None):
    """Render chat conversation history and question input."""
    st.subheader("💬 Ask Your Documents")

    # Render previous conversation turns from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # User input box
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send query to FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence & generating answer..."):
                try:
                    payload = {
                        "query": prompt,
                        "doc_id": active_doc_id,
                        "stream": False,
                    }
                    response = requests.post(f"{backend_url}/api/v1/query", json=payload, timeout=45)

                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status")

                        if status == "refused":
                            answer_text = f"⚠️ **Refusal:** {data.get('reason', 'Context is insufficient to answer.')}"
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
