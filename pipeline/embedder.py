"""
pipeline/embedder.py
--------------------
Embeds code chunks using Gemini and stores them in ChromaDB.

Design decisions:
- One Chroma collection per session (wiped on each new repo).
- persist_directory = "./chroma_codebuddy" so it survives restarts.
- Embedding model: Gemini "models/embedding-001" (free tier, 768-dim).
"""

import os
import shutil
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "./chroma_codebuddy"
COLLECTION_NAME = "codebuddy"


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a Gemini embedding model instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. "
            "Add it to your .env file or Streamlit secrets."
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document",
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embed all chunks and store in Chroma.
    Wipes any existing index first (fresh start for each repo).

    Args:
        chunks: Output of splitter.split_documents()

    Returns:
        Chroma vector store ready for similarity search.
    """
    # Wipe old index so we start fresh
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"[Embedder] Cleared old index at {DB_DIR}")

    print(f"[Embedder] Embedding {len(chunks)} chunks with Gemini ...")
    print(f"[Embedder] This may take 1–3 minutes for large repos ...")

    embeddings = _get_embeddings()

    # Chroma.from_documents embeds + indexes in one call
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    count = vectordb._collection.count()
    print(f"[Embedder] Done. {count} vectors stored in {DB_DIR}")
    return vectordb


def load_vectorstore() -> Chroma:
    """
    Load an existing Chroma index from disk.
    Use this to avoid re-embedding on every Streamlit reload.

    Raises FileNotFoundError if no index exists yet.
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"No index found at {DB_DIR}. "
            "Index a repo first by clicking 'Index repo'."
        )

    print(f"[Embedder] Loading existing index from {DB_DIR}")
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def add_documents(vectordb: Chroma, new_chunks: list[Document]) -> Chroma:
    """
    Incrementally add new chunks to an existing vector store.
    Useful when a user wants to add a second repo to the same index.
    """
    vectordb.add_documents(new_chunks)
    print(f"[Embedder] Added {len(new_chunks)} new chunks to existing index.")
    return vectordb