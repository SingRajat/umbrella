import streamlit as st
from frontend.components.chat import render_chat_interface
from frontend.components.upload import render_upload_widget

st.set_page_config(page_title="Umbrella — Workspace", page_icon="☂️", layout="wide")

st.title("☂️ RAG Workspace")

# 2-Column layout: Left column for document ingestion and management, Right column for chat
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    render_upload_widget()

with right_col:
    render_chat_interface()
