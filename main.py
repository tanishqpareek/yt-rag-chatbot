from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


# step 1 - get the transcript from youtube
def fetch_transcript(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        # joining all caption pieces into one big string
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print(f"got transcript — {len(transcript)} characters")
        return transcript
    except TranscriptsDisabled:
        raise ValueError("this video has no captions, try another one")
    except Exception as e:
        raise ValueError(f"couldn't fetch transcript: {e}")


# step 2 - split the big transcript into smaller pieces
# i tried different chunk sizes, 1000 with 200 overlap worked best for me
def create_chunks(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])
    print(f"split into {len(chunks)} chunks")
    return chunks


# step 3 - convert chunks into vectors and store them
# using miniLM because its free and works well enough
def build_vector_store(chunks):
    print("building vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("vector store ready")
    return vector_store


# step 4 - the actual RAG pipeline
def build_rag_chain(vector_store):

    # grab top 4 most relevant chunks for any question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # telling the llm to only use the transcript, not its own knowledge
    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions about a YouTube video.

Rules:
- Only use the transcript context below to answer
- If the answer isn't there, just say you don't know
- Keep it short and clear

Transcript context:
{context}

Question: {question}

Answer:
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2
    )

    parser = StrOutputParser()

    # running retrieval and question in parallel then combining
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain


# step 5 - cli version for running directly in terminal
def main():
    print("YouTube RAG Chatbot")
    print("-------------------")

    video_id = input("enter video id: ").strip()

    if not video_id:
        print("no video id entered, exiting")
        return

    transcript = fetch_transcript(video_id)
    chunks = create_chunks(transcript)
    vector_store = build_vector_store(chunks)
    chain = build_rag_chain(vector_store)

    print("\nready! type your questions below")
    print("type exit to quit\n")

    while True:
        question = input("you: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("bye!")
            break

        answer = chain.invoke(question)
        print(f"bot: {answer}\n")


if __name__ == "__main__":
    main()
