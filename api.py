"""
YouTube RAG Chatbot — FastAPI REST API
Author: Tanishq Pareek

Run with: uvicorn api:app --reload
Docs at:  http://localhost:8000/docs
"""

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
    Fetches transcript, chunks it, and builds the vector store.
    Must call this before /ask.
    """
    video_id = request.video_id.strip()

    if video_id in loaded_videos:
        return LoadVideoResponse(
            message="Video already loaded. Ready to answer questions.",
            video_id=video_id,
            chunks_created=loaded_videos[video_id]["chunk_count"]
        )

    try:
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
