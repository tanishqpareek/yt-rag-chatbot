"""
YouTube RAG Chatbot — Streamlit Frontend
Author: Tanishq Pareek
Run with: streamlit run streamlit_app.py
"""

import streamlit as st

# ── Page Config (must be the first Streamlit command) ──
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="centered"
)

import requests
import re
import hashlib
import threading
import time
import uvicorn

API_BASE = "http://localhost:8000"


@st.cache_resource(show_spinner="Starting the app, the first load takes about a minute...")
def start_api():
    import api  # your api.py
    config = uvicorn.Config(api.app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(3)  # give the server a moment to start
    return True


start_api()
