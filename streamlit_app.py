"""
YouTube RAG Chatbot — Streamlit Frontend
Author: Tanishq Pareek
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import re
import hashlib
import threading
import time
import uvicorn

API_BASE = "http://localhost:8000"


@st.cache_resource
def start_api():
    import api  # your api.py
    config = uvicorn.Config(api.app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(3)  # give the server a moment to start
    return True


start_api()

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f0f; color: #f1f1f1; }

    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #f1f1f1;
        border: 1px solid #333;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #ff0000;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #cc0000; }

    /* Chat messages */
    .stChatMessage { background-color: #1a1a1a; border-radius: 10px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }

    /* Status box */
    .status-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-left: 3px solid #ff0000;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 13px;
        color: #aaa;
    }

    /* Chunk badge */
    .chunk-badge {
        display: inline-block;
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 12px;
        color: #ff6666;
        margin-top: 6px;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        color: #f1f1f1;
    }
    .main-header p {
        color: #888;
        font-size: 14px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: extract video ID from URL or raw ID ──
def extract_video_id(input_str: str) -> str:
    input_str = input_str.strip()
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", input_str):
        return input_str
