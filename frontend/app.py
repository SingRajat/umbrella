"""Streamlit application entry point and navigation setup."""
import os
import requests
import streamlit as st


def get_backend_url() -> str:
    """Resolve backend API URL from environment variable with local fallback."""
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def check_backend_health(backend_url: str) -> dict:
    """Ping backend health endpoint to check service status."""
    try:
        response = requests.get(f"{backend_url}/api/v1/health", timeout=3)
        if response.status_code == 200:
            return {"status": "connected", "data": response.json()}
        return {"status": "error", "message": f"Status {response.status_code}"}
    except Exception as exc:
        return {"status": "disconnected", "message": str(exc)}


def main():
    st.set_page_config(
        page_title="Umbrella — RAG Pipeline",
        page_icon="☂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize in-memory session states if not already set
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None

    backend_url = get_backend_url()
    health = check_backend_health(backend_url)

    st.sidebar.title("☂️ Umbrella RAG")
    if health["status"] == "connected":
        st.sidebar.success("Backend: Connected ✅")
    else:
        st.sidebar.warning(f"Backend: {health['status'].capitalize()} ⚠️")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pages**\n"
        "- **1. Landing**: System overview & workflow\n"
        "- **2. Workspace**: Upload & query documents\n"
        "- **3. Evaluation**: RAGAS experiment metrics"
    )

    st.title("Welcome to Umbrella ☂️")
    st.markdown(
        "A modular, production-oriented Retrieval-Augmented Generation (RAG) system with measurable evaluation."
    )
    st.info("👈 Use the sidebar to navigate between **Landing**, **Workspace**, and **Evaluation**.")


if __name__ == "__main__":
    main()
