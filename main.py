"""
AI-Powered YouTube RAG Chatbot
Author: Tanishq Pareek
GitHub: github.com/tanishqpareek
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ─────────────────────────────────────────────
# STEP 1: Fetch YouTube Transcript
# ─────────────────────────────────────────────

def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript from a YouTube video using its video ID.
    Example video ID: for https://youtube.com/watch?v=abc123 → video_id = "abc123"
    """
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print(f"✅ Transcript fetched — {len(transcript)} characters")
        return transcript
    except TranscriptsDisabled:
        raise ValueError("❌ No captions available for this video.")
    except Exception as e:
        raise ValueError(f"❌ Error fetching transcript: {e}")


# ─────────────────────────────────────────────
# STEP 2: Split Transcript into Chunks
# ─────────────────────────────────────────────

def create_chunks(transcript: str):
    """
    Split the transcript into overlapping chunks.
    - chunk_size=1000  : each chunk is max 1000 characters
    - chunk_overlap=200: consecutive chunks share 200 characters
      to avoid losing context at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# STEP 3: Create Embeddings + Vector Store
# ─────────────────────────────────────────────

def build_vector_store(chunks):
    """
    Convert each chunk into a vector (embedding) and store in FAISS.
    Uses HuggingFace's free MiniLM model — no API key needed.
    """
    print("⏳ Building vector store (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store built")
    return vector_store


# ─────────────────────────────────────────────
# STEP 4: Build the RAG Chain
# ─────────────────────────────────────────────

def build_rag_chain(vector_store):
    """
    Build the full RAG pipeline using LangChain LCEL:
    Question → Retrieve top 4 chunks → Format context →
    Fill prompt → LLM → Parse output string
    """

    # Retriever: finds top 4 most similar chunks for any question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Format retrieved documents into a single context string
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Prompt template — constrains LLM to answer only from context
    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions about a YouTube video.

IMPORTANT RULES:
- Answer ONLY using the transcript context provided below.
- If the answer is not in the context, say: "I don't have that information in this video."
- Be concise and clear.
- Do not make up information.

Context (from video transcript):
{context}

Question: {question}

Answer:
        """,
        input_variables=["context", "question"]
    )

    # LLM — using Google Gemini (free tier)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.2   # low temperature = factual, consistent answers
    )

    # Output parser — extracts plain string from LLM response
    parser = StrOutputParser()

    # Build parallel chain:
    # Same question goes to BOTH branches simultaneously:
    # Branch 1: question → retriever → format_docs → context string
    # Branch 2: question passes through unchanged
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    # Final chain: parallel → prompt → LLM → parse
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain


# ─────────────────────────────────────────────
# STEP 5: Main Application
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   🎬 YouTube RAG Chatbot — by Tanishq Pareek")
    print("=" * 55)

    # Get video ID from user
    print("\nEnter the YouTube video ID.")
    print("Example: for https://youtube.com/watch?v=abc123")
    print("         enter just: abc123\n")
    video_id = input("Video ID: ").strip()

    if not video_id:
        print("❌ No video ID entered. Exiting.")
        return

    # Build the pipeline
    print("\n📥 Fetching transcript...")
    transcript = fetch_transcript(video_id)

    print("✂️  Splitting into chunks...")
    chunks = create_chunks(transcript)

    print("🔍 Building vector store...")
    vector_store = build_vector_store(chunks)

    print("⚙️  Building RAG chain...")
    chain = build_rag_chain(vector_store)

    print("\n✅ Ready! Ask any question about the video.")
    print("   Type 'exit' or 'quit' to stop.\n")
    print("-" * 55)

    # Chat loop
    while True:
        question = input("\n🙋 You: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        print("🤖 Bot: ", end="", flush=True)
        answer = chain.invoke(question)
        print(answer)
        print("-" * 55)


if __name__ == "__main__":
    main()
