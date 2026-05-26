"""
YouTube RAG Chatbot — Streamlit Frontend
Author: Tanishq Pareek
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import re

API_BASE = "http://localhost:8000"

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
    return None


# ── Session State Init ───────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_video_id" not in st.session_state:
    st.session_state.loaded_video_id = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 Load a YouTube Video")
    st.markdown("<div style='color:#666; font-size:13px; margin-bottom:12px'>Paste a YouTube URL or video ID</div>", unsafe_allow_html=True)

    video_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=... or dQw4w9WgXcQ",
        label_visibility="collapsed"
    )

    load_clicked = st.button("Load Video", use_container_width=True)

    if load_clicked and video_input:
        video_id = extract_video_id(video_input)

        if not video_id:
            st.error("Could not extract a valid video ID. Check your URL.")
        else:
            with st.spinner("Fetching transcript and building index..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/load",
                        json={"video_id": video_id},
                        timeout=60
                    )
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        st.session_state.loaded_video_id = video_id
                        st.session_state.chunk_count = data["chunks_created"]
                        st.session_state.messages = []
                        st.success(data["message"])
                    else:
                        st.error(resp.json().get("detail", "Failed to load video."))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure FastAPI is running on localhost:8000")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ── Loaded Video Status ──
    if st.session_state.loaded_video_id:
        st.markdown(f"""
        <div class='status-box'>
            ✅ <strong>Video Loaded</strong><br>
            <span style='font-family:monospace'>{st.session_state.loaded_video_id}</span><br>
            <span class='chunk-badge'>{st.session_state.chunk_count} chunks indexed</span>
        </div>
        """, unsafe_allow_html=True)

        # Thumbnail preview
        st.image(
            f"https://img.youtube.com/vi/{st.session_state.loaded_video_id}/mqdefault.jpg",
            use_column_width=True
        )

        # Unload button
        if st.button("Unload Video", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/unload/{st.session_state.loaded_video_id}", timeout=10)
            except Exception:
                pass
            st.session_state.loaded_video_id = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='color:#555; font-size:12px; text-align:center'>Built by Tanishq Pareek</div>", unsafe_allow_html=True)


# ── Main Area ────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🎬 YouTube RAG Chatbot</h1>
    <p>Load any YouTube video and ask questions about its content</p>
</div>
""", unsafe_allow_html=True)

# No video loaded state
if not st.session_state.loaded_video_id:
    st.markdown("""
    <div style='text-align:center; margin-top:80px; color:#444'>
        <div style='font-size:48px'>📺</div>
        <div style='font-size:16px; margin-top:16px'>Paste a YouTube URL in the sidebar to get started</div>
        <div style='font-size:13px; margin-top:8px; color:#333'>Supports any video with captions/subtitles</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat History ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # ── Chat Input ──
    question = st.chat_input("Ask anything about the video...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        # Get answer from API
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ask",
                        json={
                            "video_id": st.session_state.loaded_video_id,
                            "question": question
                        },
                        timeout=30
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    elif resp.status_code == 404:
                        err = "Video not loaded on the server. Please reload it."
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                    else:
                        err = resp.json().get("detail", "Something went wrong.")
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})

                except requests.exceptions.ConnectionError:
                    err = "Lost connection to API. Make sure FastAPI is running."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"Unexpected error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

    # ── Clear Chat ──
    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=False):
            st.session_state.messages = []
            st.rerun()
