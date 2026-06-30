# 🤖 CodeBuddy

**Chat with any GitHub repository.** CodeBuddy clones a public repo, indexes its source code using a RAG (Retrieval-Augmented Generation) pipeline, and lets you ask natural-language questions about the codebase — with answers grounded in the actual files and cited by filename.

🔗 **Live demo:** [codebuddyai.streamlit.app](https://codebuddyai.streamlit.app)

---

## ✨ Features

- 📥 **One-click repo indexing** — paste any public GitHub URL and CodeBuddy shallow-clones and processes it
- 🧠 **Language-aware chunking** — code is split on natural boundaries (functions, classes) instead of arbitrary character counts, using dedicated splitters for Python, JavaScript, Java, Go, Rust, and more
- 🔍 **MMR retrieval** — Max Marginal Relevance retrieval returns diverse, non-redundant code chunks instead of near-duplicate matches
- 📄 **Source-cited answers** — every response references the exact file(s) it came from
- 💬 **Conversational chat UI** — ask follow-up questions, get code blocks, explanations, and architecture overviews
- 📊 **Index stats** — see chunk counts and language breakdown after indexing
- ⚙️ **Configurable chunking** — adjust chunk size, overlap, and retrieval depth from the sidebar

---

## 🏗️ Architecture

```
GitHub URL
    │
    ▼
┌─────────────┐
│   Loader    │  Shallow clone (depth=1) → walk files → filter by extension
└──────┬──────┘
       ▼
┌─────────────┐
│  Splitter   │  Language-aware chunking (Python/JS/Java/Go/Rust splitters)
└──────┬──────┘
       ▼
┌─────────────┐
│  Embedder   │  Gemini embeddings → stored in ChromaDB
└──────┬──────┘
       ▼
┌─────────────┐
│  Retriever  │  MMR search (k=6, fetch_k=20) — diverse, relevant chunks
└──────┬──────┘
       ▼
┌─────────────┐
│    Chain    │  Prompt + context + question → Gemini 2.5 Flash
└──────┬──────┘
       ▼
   Answer + cited sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding (`models/gemini-embedding-001`) |
| Orchestration | LangChain (LCEL) |
| Vector store | ChromaDB |
| Frontend | Streamlit |
| Repo cloning | GitPython |

---

## 📁 Project Structure

```
codebuddy/
├── app.py                  # Streamlit UI — sidebar, chat, indexing flow
├── pipeline/
│   ├── loader.py            # Clones repo, loads files into Documents
│   ├── splitter.py          # Language-aware chunking
│   ├── embedder.py          # Gemini embeddings → ChromaDB
│   └── chain.py             # RAG chain (retriever + prompt + LLM)
├── requirements.txt
└── .env                     
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier available)
- Git installed locally

### Installation

```bash
# Clone this repo
git clone https://github.com/Khushirajput123/codebuddy.git
cd codebuddy

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 💡 Usage

1. Paste a public GitHub repository URL into the sidebar
2. (Optional) Adjust chunk size, overlap, and retrieval depth under **Advanced settings**
3. Click **🚀 Index Repository** and wait for indexing to complete
4. Ask questions in the chat, such as:
   - *"What does this codebase do overall?"*
   - *"Where is the main entry point?"*
   - *"How does authentication work?"*
   - *"Explain the `OrderService` class."*
   - *"Where is the database connection set up?"*

Each answer includes a **Sources** panel showing exactly which files were used to generate it.

---

## ⚙️ Configuration

These are adjustable from the sidebar's **Advanced settings** panel:

| Setting | Default | Description |
|---|---|---|
| Chunk size | 1000 chars | Larger = more context per chunk, fewer chunks |
| Chunk overlap | 150 chars | Preserves context across chunk boundaries |
| Top-k retrieval | 6 | Number of chunks sent to the LLM per question |

---
## 🤝 Contributing

Issues and pull requests are welcome. If you find a bug or have a feature idea, feel free to open an issue.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

