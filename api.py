Claude
Added to memory youtube-rag-chatbot.md
Your main.py is clean, with no key in it and the key read from the environment. There are four small changes, in this order. Edit each file on GitHub with the pencil icon, then commit.

1. main.py: check saved transcripts first
At the top, add this line next to your other imports:

python
from pathlib import Path
Then replace your whole fetch_transcript function with this:

python
SAVED_DIR = Path(__file__).parent / "transcripts"

def fetch_transcript(video_id: str) -> str:
    # use a saved transcript first (YouTube blocks cloud servers)
    saved = SAVED_DIR / f"{video_id}.txt"
    if saved.exists():
        transcript = saved.read_text(encoding="utf-8")
        print(f"loaded saved transcript — {len(transcript)} characters")
        return transcript
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print(f"got transcript — {len(transcript)} characters")
        return transcript
    except TranscriptsDisabled:
        raise ValueError("this video has no captions, try another one")
    except Exception as e:
        raise ValueError(f"couldn't fetch transcript: {e}")
2. api.py: accept a pasted transcript
Add this import at the top:

python
from typing import Optional
Replace the LoadVideoRequest class with:

python
class LoadVideoRequest(BaseModel):
    video_id: str
    transcript: Optional[str] = None
Inside load_video, replace the line transcript = fetch_transcript(video_id) with:

python
        if request.transcript and request.transcript.strip():
            transcript = request.transcript.strip()
        else:
            transcript = fetch_transcript(video_id)
3. streamlit_app.py: add the paste box
Add import hashlib at the top. Then find the line # ── Loaded Video Status ── and paste this just above it (same indentation as the if load_clicked line, 4 spaces):

python
    with st.expander("Video not loading? Paste the transcript"):
        st.caption("On YouTube: click ...more under the video, then Show transcript. Copy the text and paste it here.")
        pasted = st.text_area("Transcript", height=150, label_visibility="collapsed",
                              placeholder="Paste the transcript here...")
        if st.button("Load pasted transcript", use_container_width=True) and pasted.strip():
            pid = "pasted-" + hashlib.md5(pasted.encode()).hexdigest()[:8]
            with st.spinner("Building index..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/load",
                        json={"video_id": pid, "transcript": pasted},
                        timeout=60
                    )
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        st.session_state.loaded_video_id = pid
                        st.session_state.chunk_count = data["chunks_created"]
                        st.session_state.messages = []
                        st.success(data["message"])
                    else:
                        st.error(resp.json().get("detail", "Failed to load transcript."))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
Then find the st.image( block for the thumbnail and indent it under this condition, because a pasted transcript has no thumbnail:

python
        if not st.session_state.loaded_video_id.startswith("pasted-"):
            st.image(
                f"https://img.youtube.com/vi/{st.session_state.loaded_video_id}/mqdefault.jpg",
                use_container_width=True
            )
4. Save demo transcripts (on your laptop)
Create a file called save_transcripts.py in your project folder with this:

python
import sys
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

out = Path("transcripts")
out.mkdir(exist_ok=True)

for vid in sys.argv[1:]:
    try:
        parts = YouTubeTranscriptApi().fetch(vid)
        text = " ".join(p.text for p in parts)
        (out / f"{vid}.txt").write_text(text, encoding="utf-8")
        print(f"saved {vid} ({len(text)} chars)")
    except Exception as e:
        print(f"failed {vid}: {e}")
Open a terminal in that folder and run it with 3 or 4 video IDs. The ID is the 11 characters after v= in the link:

python save_transcripts.py ID1 ID2 ID3
Then on GitHub, click Add file > Upload files, drag the new transcripts folder in, and commit. Check that the .txt files show up in the repo.

Streamlit redeploys on its own after each commit. Wait a minute or two, open your live link, and load one of the saved videos. Tell me what happens. Note the sample video IDs somewhere, because I’d also like to add them to the app as “try this” examples.

i dont have any transcripts on my laptop

Offering two ways to get transcripts into GitHub
You don’t need to have them already. The script was only a way to download them, and there’s an easier way that needs nothing on your laptop. You’ll copy each transcript from YouTube and save it straight into GitHub.

For each of 3 or 4 videos

Open the video on YouTube. Pick short ones (5 to 15 minutes) with English captions.
Under the video, click ...more, then scroll down and click Show transcript.
In the transcript panel, click the three dots at the top and choose Toggle timestamps so the numbers disappear.
Click once in the transcript text, press Ctrl+A to select all, then Ctrl+C to copy.
Look at the video’s link and find the 11 characters after v=. That’s the video ID. Write it down.
Go to your repo on GitHub (github.com/tanishqpareek/yt-rag-chatbot) and click Add file > Create new file.
In the filename box, type transcripts/ followed by the ID and .txt, for example transcripts/AbCdEfGhIjK.txt. Typing the slash creates the folder. The ID must match the link exactly, including capital letters.
Paste the transcript into the big text area with Ctrl+V.
Click Commit changes, then Commit changes again.
Repeat for each video. Then wait a minute for Streamlit to redeploy, open your live app, paste one of those video links, and click Load Video.

Before this works
Your edits to main.py, api.py and streamlit_app.py from my last message need to be committed. Did you finish those? The saved transcripts only work with the new fetch_transcript in main.py.

Tell me the 3 or 4 video IDs you save and I’ll add them to the app as “try these” examples.

i dont know what to replace in api.py its confusing

Easiest fix: replace the whole file instead of making small edits.

On GitHub, open api.py and click the pencil icon (top right).
Click inside the text, press Ctrl+A to select everything, and press Delete.
Paste everything below.
Click Commit changes, then Commit changes again.
python
"""
YouTube RAG Chatbot — FastAPI REST API
Author: Tanishq Pareek

Run with: uvicorn api:app --reload
Docs at:  http://localhost:8000/docs
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import fetch_transcript, create_chunks, build_vector_store, build_rag_chain

app = FastAPI(
    title="YouTube RAG Chatbot API",
    description="Ask questions about any YouTube video using RAG — by Tanishq Pareek",
    version="1.0.0"
)

# In-memory store: video_id → rag_chain
# (in production, use a persistent cache like Redis)
loaded_videos: dict = {}


# ─── Request / Response Models ───────────────

class LoadVideoRequest(BaseModel):
    video_id: str
    transcript: Optional[str] = None

class AskRequest(BaseModel):
    video_id: str
    question: str

class LoadVideoResponse(BaseModel):
    message: str
    video_id: str
    chunks_created: int

class AskResponse(BaseModel):
    video_id: str
    question: str
    answer: str


# ─── Endpoints ───────────────────────────────

@app.get("/")
def root():
    return {
        "message": "YouTube RAG Chatbot API is running!",
        "author": "Tanishq Pareek",
        "docs": "/docs"
    }


@app.post("/load", response_model=LoadVideoResponse, status_code=201)
def load_video(request: LoadVideoRequest):
    """
    Load a YouTube video by its ID.
    Uses a pasted transcript if given, otherwise a saved or fetched one.
    Chunks it and builds the vector store. Must call this before /ask.
    """
    video_id = request.video_id.strip()

    if video_id in loaded_videos:
        return LoadVideoResponse(
            message="Video already loaded. Ready to answer questions.",
            video_id=video_id,
            chunks_created=loaded_videos[video_id]["chunk_count"]
        )

    try:
        if request.transcript and request.transcript.strip():
            transcript = request.transcript.strip()
        else:
            transcript = fetch_transcript(video_id)

        chunks = create_chunks(transcript)
        vector_store = build_vector_store(chunks)
        chain = build_rag_chain(vector_store)

        loaded_videos[video_id] = {
            "chain": chain,
            "chunk_count": len(chunks)
        }

        return LoadVideoResponse(
            message="Video loaded successfully. You can now ask questions.",
            video_id=video_id,
            chunks_created=len(chunks)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Ask a question about a previously loaded YouTube video.
    Returns a grounded answer from the video transcript.
    """
    video_id = request.video_id.strip()
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if video_id not in loaded_videos:
        raise HTTPException(
            status_code=404,
            detail=f"Video '{video_id}' not loaded. Call POST /load first."
        )

    try:
        chain = loaded_videos[video_id]["chain"]
        answer = chain.invoke(question)

        return AskResponse(
            video_id=video_id,
            question=question,
            answer=answer
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")


@app.delete("/unload/{video_id}")
def unload_video(video_id: str):
    """
    Remove a loaded video from memory to free resources.
    """
    if video_id not in loaded_videos:
        raise HTTPException(status_code=404, detail="Video not found in memory.")

    del loaded_videos[video_id]
    return {"message": f"Video '{video_id}' unloaded successfully."}


@app.get("/loaded")
def list_loaded_videos():
    """
    List all currently loaded video IDs.
    """
    return {
        "loaded_videos": list(loaded_videos.keys()),
        "count": len(loaded_videos)
    }
The only differences from your old file are the from typing import Optional line, the transcript: Optional[str] = None line in LoadVideoRequest, and the if request.transcript...else block inside load_video.

Tell me when it’s committed. Do you want the full main.py and streamlit_app.py the same way, so you can copy-paste those too?





Claude is AI and can make mistakes. Please double-check responses.






Claude finished the response
