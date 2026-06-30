"""
pipeline/chain.py
-----------------
Builds the LangChain RetrievalQA chain with:
  - MMR retriever (diversity over pure similarity)
  - Code-aware prompt (always cite filename + function)
  - Gemini 2.5 Flash as the LLM
"""

import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── Prompt template ──────────────────────────────────────────────────
CODE_QA_PROMPT = PromptTemplate(
    template="""You are CodeBuddy — an expert software engineer helping a
developer understand an unfamiliar codebase.

STRICT RULES:
1. Answer ONLY using the code context provided below.
2. ALWAYS mention which file(s) the answer comes from.
3. When describing a function or class, show its signature.
4. Use code blocks (```language) for any code you reference.
5. If the answer is not in the context, say exactly:
   "This wasn't found in the indexed files. Try rephrasing or check if the file was indexed."
6. Do NOT make up code that isn't in the context.
7. Be concise — developers want precise answers, not essays.

── Code Context ────────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────────────────

Developer Question: {question}

Answer (cite filename, show code if relevant):""",
    input_variables=["context", "question"],
)

# ── Sample questions shown in the UI ────────────────────────────────
SAMPLE_QUESTIONS = [
    "What does this codebase do overall?",
    "Where is the main entry point of this application?",
    "How does authentication / login work?",
    "Where is the database connection set up?",
    "What are all the API endpoints / routes?",
    "How is error handling done across the codebase?",
    "Explain what the [ClassName] class does.",
    "Where is configuration / environment variables loaded?",
    "What testing framework is used and where are the tests?",
    "How does data flow from input to output in this app?",
]


def _format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(
        f"# File: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def build_chain(vectordb: Chroma):
    """
    Build and return the RAG chain using LCEL (LangChain Expression Language).

    Retriever: MMR (Max Marginal Relevance)
        - fetch_k=20  : consider 20 candidate chunks
        - k=6         : return 6 most diverse + relevant ones

    LLM: gemini-2.5-flash
        - temperature=0 : deterministic, no hallucination guessing
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )

    # MMR retriever — diverse chunks, avoids duplicates
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

    # LCEL chain — modern replacement for RetrievalQA
    chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | CODE_QA_PROMPT
            | llm
            | StrOutputParser()
    )

    return chain, retriever  # return both so app.py can get source docs


def format_sources(source_docs: list) -> list[dict]:
    """
    Extract clean source info from retrieved Documents.
    Returns list of {file, language, chunk_index} dicts.
    De-duplicated by filename.
    """
    seen = set()
    out = []
    for doc in source_docs:
        meta = doc.metadata
        fname = meta.get("source", "unknown")
        if fname in seen:
            continue
        seen.add(fname)
        out.append({
            "file": fname,
            "language": meta.get("language", "text"),
            "size_kb": meta.get("size_kb", "?"),
            "preview": doc.page_content[:120].replace("\n", " "),
        })
    return out