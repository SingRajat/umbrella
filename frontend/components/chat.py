import httpx
import streamlit as st
from frontend.components.citations import render_citations_panel

API_BASE_URL = "http://127.0.0.1:8000/api/v1"


def render_chat_interface():
    """Renders session-based conversation interface with citation markers and refusal handling."""
    st.subheader("💬 RAG Question Answering")

    # Initialize in-memory session conversation history per user
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past conversation turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                render_citations_panel(msg["citations"])
            if msg.get("status") == "refused":
                st.warning(f"⚠️ **Refusal Reason:** `{msg.get('reason')}`")

    # Question Input
    prompt = st.chat_input("Ask a question about your ingested documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and formulating grounded answer..."):
                try:
                    payload = {"query": prompt, "stream": False}
                    res = httpx.post(
                        f"{API_BASE_URL}/query",
                        json=payload,
                        timeout=60.0,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        status = data.get("status")

                        if status == "answered":
                            answer = data.get("answer", "")
                            citations = data.get("citations", [])

                            st.markdown(answer)
                            if citations:
                                render_citations_panel(citations)

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "status": "answered",
                                    "citations": citations,
                                }
                            )
                        elif status == "refused":
                            reason = data.get("reason", "insufficient_context")
                            refusal_msg = "I cannot answer this question because the provided documents do not contain sufficient evidence."
                            st.markdown(refusal_msg)
                            st.warning(f"⚠️ **Refusal Triggered:** `{reason}`")

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": refusal_msg,
                                    "status": "refused",
                                    "reason": reason,
                                    "citations": [],
                                }
                            )
                    elif res.status_code == 429:
                        st.error("Rate limit exceeded. Please wait a moment before sending more queries.")
                    else:
                        st.error(f"Backend returned error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with backend: {e}")
