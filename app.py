




"""
app.py
------
CodeBuddy — Chat with any GitHub repository.

Run:  streamlit run app.py
"""

import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from pipeline.loader import load_repo
from pipeline.splitter import split_documents, get_file_stats
from pipeline.embedder import build_vectorstore, load_vectorstore
from pipeline.chain import build_chain, format_sources, SAMPLE_QUESTIONS

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeBuddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .source-badge {
      display: inline-block;
      background: #1e2230;
      color: #a8b2c8;
      font-family: monospace;
      font-size: 12px;
      padding: 3px 10px;
      border-radius: 4px;
      margin: 3px 3px 3px 0;
      border: 1px solid #2d3348;
  }
  .stat-box {
      background: #0e1117;
      border: 1px solid #2d3348;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 8px;
  }
  .ready-banner {
      background: #0a2e1e;
      border: 1px solid #1a6b3c;
      border-radius: 8px;
      padding: 10px 14px;
      color: #4ecca3;
      font-size: 14px;
      margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────
def init_state():
    defaults = {
        "vectordb": None,
        "chain": None,
        "retriever": None,
        "messages": [],
        "repo_url": "",
        "indexed": False,
        "file_stats": {},
        "total_chunks": 0,
        "tmp_dir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 CodeBuddy")
    st.caption("Chat with any public GitHub repository")

    st.markdown("---")

    # ── Repo URL input ───────────────────────────────────────────────
    st.subheader("📦 Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/username/repo",
        value=st.session_state.repo_url,
        help="Must be a public repository.",
    )

    # ── Chunking settings ────────────────────────────────────────────
    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider(
            "Chunk size (chars)",
            min_value=300, max_value=2000,
            value=1000, step=100,
            help="Larger = more context per chunk. Smaller = more precise retrieval.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (chars)",
            min_value=0, max_value=400,
            value=150, step=50,
            help="Overlap between adjacent chunks to preserve context at boundaries.",
        )
        top_k = st.slider(
            "Chunks to retrieve (k)",
            min_value=2, max_value=10,
            value=6, step=1,
            help="How many code chunks to send to the LLM per question.",
        )

    # ── Index button ─────────────────────────────────────────────────
    index_btn = st.button(
        "🚀 Index Repository",
        type="primary",
        use_container_width=True,
        disabled=not repo_url,
    )

    # ── Stats (shown after indexing) ─────────────────────────────────
    if st.session_state.indexed:
        st.markdown("---")
        st.subheader("📊 Index Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total chunks", st.session_state.total_chunks)
        with col2:
            st.metric("Languages", len(st.session_state.file_stats))

        st.markdown("**Languages indexed:**")
        for lang, count in st.session_state.file_stats.items():
            st.markdown(
                f'<div class="stat-box"><b>{lang}</b> — {count} chunks</div>',
                unsafe_allow_html=True,
            )

        if st.button("🗑️ Clear index & start over", use_container_width=True):
            if st.session_state.tmp_dir and os.path.exists(st.session_state.tmp_dir):
                shutil.rmtree(st.session_state.tmp_dir, ignore_errors=True)
            st.session_state.vectordb = None
            st.session_state.chain = None
            st.session_state.retriever = None
            st.session_state.messages = []
            st.session_state.indexed = False
            st.session_state.file_stats = {}
            st.session_state.total_chunks = 0
            st.session_state.tmp_dir = None
            st.session_state.repo_url = ""
            st.rerun()

    st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# INDEXING LOGIC (runs when button clicked)
# ════════════════════════════════════════════════════════════════════
if index_btn and repo_url:
    st.session_state.messages = []  # clear old chat
    st.session_state.repo_url = repo_url
    st.session_state.indexed = False

    with st.status("🔄 Indexing repository...", expanded=True) as status:

        # Step 1: Clone
        st.write("📥 **Step 1/4** — Cloning repository...")
        try:
            docs, tmp_dir = load_repo(repo_url)
            st.session_state.tmp_dir = tmp_dir
            st.write(f"✅ Loaded **{len(docs)}** files from the repository.")
        except ValueError as e:
            status.update(label="❌ Clone failed", state="error")
            st.error(str(e))
            st.stop()

        if not docs:
            status.update(label="❌ No files found", state="error")
            st.error("No readable code files found in this repo. "
                     "Make sure it's a public repo with source code.")
            st.stop()

        # Step 2: Split
        st.write("✂️  **Step 2/4** — Splitting into chunks...")
        chunks = split_documents(docs, chunk_size, chunk_overlap)
        st.write(f"✅ Created **{len(chunks)}** chunks.")

        if not chunks:
            status.update(label="❌ Split failed", state="error")
            st.error("No chunks were created. Try increasing the chunk size.")
            st.stop()

        # Step 3: Embed + index
        st.write("🧠 **Step 3/4** — Embedding with Gemini (may take 1–3 min)...")
        try:
            vectordb = build_vectorstore(chunks)
            st.session_state.vectordb = vectordb
        except Exception as e:
            status.update(label="❌ Embedding failed", state="error")
            st.error(f"Embedding error: {e}")
            st.stop()
        st.write(f"✅ Indexed **{vectordb._collection.count()}** vectors in Chroma.")

        # Step 4: Build chain
        st.write("⛓️  **Step 4/4** — Building QA chain...")
        chain, retriever = build_chain(vectordb)
        st.session_state.chain = chain
        st.session_state.retriever = retriever

        # Save stats
        st.session_state.file_stats = get_file_stats(chunks)
        st.session_state.total_chunks = len(chunks)
        st.session_state.indexed = True

        status.update(label="✅ Ready! Ask your first question below.", state="complete")

    # Force a fresh rerun so chat UI renders immediately after indexing
    st.rerun()


# ════════════════════════════════════════════════════════════════════
# HELPER: run one question through the chain, return (answer, sources)
# ════════════════════════════════════════════════════════════════════
def run_question(question: str):
    try:
        answer = st.session_state.chain.invoke(question)
        source_docs = st.session_state.retriever.invoke(question)
        sources = format_sources(source_docs)
    except Exception as e:
        answer = f"❌ Error: {e}\n\nTry rephrasing your question."
        sources = []
    return answer, sources


# ════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ════════════════════════════════════════════════════════════════════
if not st.session_state.indexed:
    # ── Welcome screen ───────────────────────────────────────────────
    st.title("🤖 CodeBuddy")
    st.markdown("### Chat with any GitHub repository")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📥 Clone**\nPaste a GitHub URL and click *Index Repository*. CodeBuddy shallow-clones the repo.")
    with col2:
        st.info("**🧠 Index**\nFiles are split with a language-aware splitter and embedded using Gemini into ChromaDB.")
    with col3:
        st.info("**💬 Chat**\nAsk anything about the codebase. Get answers with file citations and code snippets.")

    st.markdown("---")
    st.markdown("#### 💡 Example questions you can ask")
    for q in SAMPLE_QUESTIONS[:6]:
        st.markdown(f"- {q}")

    st.markdown("---")
    st.markdown(
        "> **Tip:** Try indexing a small public repo first — "
        "e.g. `https://github.com/pallets/click` or `https://github.com/tiangolo/fastapi`"
    )

else:
    # ── Ready banner ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="ready-banner">✅ <b>{st.session_state.repo_url}</b> is indexed '
        f'and ready — {st.session_state.total_chunks} chunks across '
        f'{len(st.session_state.file_stats)} language(s).</div>',
        unsafe_allow_html=True,
    )

    # ── Suggested questions as clickable chips ───────────────────────
    # Only show these BEFORE the user has asked anything.
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(3)
        for i, q in enumerate(SAMPLE_QUESTIONS[:6]):
            with cols[i % 3]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("🔍 Searching codebase..."):
                        answer, sources = run_question(q)
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources,
                    })
                    st.rerun()

    # ── Chat history (renders every past message) ───────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📁 Sources ({len(msg['sources'])} files)"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<span class="source-badge">📄 {src["file"]}</span> '
                            f'<span class="source-badge">{src["language"]}</span>',
                            unsafe_allow_html=True,
                        )
                        if src.get("preview"):
                            st.caption(f"…{src['preview']}…")

    # ════════════════════════════════════════════════════════════════
    # THE CHAT INPUT BOX — this was missing/commented out before.
    # st.chat_input() always renders pinned to the bottom of the page,
    # OUTSIDE the normal top-to-bottom flow, like a real chat app.
    # ════════════════════════════════════════════════════════════════
    if prompt := st.chat_input("Ask anything about the codebase..."):
        # 1. Show user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate + show the answer immediately
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching codebase..."):
                answer, sources = run_question(prompt)
            st.markdown(answer)
            if sources:
                with st.expander(f"📁 Sources ({len(sources)} files)"):
                    for src in sources:
                        st.markdown(
                            f'<span class="source-badge">📄 {src["file"]}</span> '
                            f'<span class="source-badge">{src["language"]}</span>',
                            unsafe_allow_html=True,
                        )
                        if src.get("preview"):
                            st.caption(f"…{src['preview']}…")

        # 3. Save BOTH messages to history so they persist after rerun
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })