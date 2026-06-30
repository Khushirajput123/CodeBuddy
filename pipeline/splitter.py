"""
pipeline/splitter.py
--------------------
Language-aware text splitting.

Key idea:
  RecursiveCharacterTextSplitter.from_language() knows each language's
  natural boundaries (class/def for Python, function for JS, etc.)
  and splits there FIRST before ever cutting mid-logic.

Fallback:
  Files with unsupported extensions (YAML, Markdown, JSON, etc.)
  get the plain RecursiveCharacterTextSplitter.
"""

from pathlib import Path
# from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain.schema import Document
from langchain_core.documents import Document

# ── Map file extension → LangChain Language enum ────────────────────
EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.JS,  # TypeScript uses same separators as JS
    ".jsx": Language.JS,
    ".tsx": Language.JS,
    ".mjs": Language.JS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,
    ".scala": Language.SCALA,
}

# ── Human-readable language name for metadata ────────────────────────
EXTENSION_TO_LANG_NAME = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React JSX", ".tsx": "React TSX", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C",
    ".h": "C Header", ".rb": "Ruby", ".php": "PHP",
    ".cs": "C#", ".kt": "Kotlin", ".swift": "Swift",
    ".md": "Markdown", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".sh": "Shell",
}


def _get_splitter(ext: str, chunk_size: int, chunk_overlap: int):
    """
    Return the right splitter for a given file extension.
    Language-aware when possible, generic recursive otherwise.
    """
    lang = EXTENSION_TO_LANGUAGE.get(ext)

    if lang:
        # Language-aware: splits on class/def/function boundaries first
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Fallback: works well for Markdown, YAML, JSON, config files etc.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_documents(
        docs: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split all loaded Documents into smaller chunks.

    For each document:
    1.  Detect the file extension.
    2.  Pick the best splitter (language-aware or generic).
    3.  Split and enrich metadata with language + chunk index.

    Returns a flat list of all chunks across all files.
    """
    all_chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix.lower()

        splitter = _get_splitter(ext, chunk_size, chunk_overlap)

        try:
            chunks = splitter.split_documents([doc])
        except Exception as e:
            print(f"[Splitter] Could not split {source}: {e}")
            continue

        # Enrich every chunk with language tag and chunk index
        lang_name = EXTENSION_TO_LANG_NAME.get(ext, "text")
        for idx, chunk in enumerate(chunks):
            chunk.metadata["language"] = lang_name
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks_in_file"] = len(chunks)

        all_chunks.extend(chunks)

    print(
        f"[Splitter] {len(docs)} files → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


def get_file_stats(chunks: list[Document]) -> dict:
    """
    Returns a summary dict useful for displaying in Streamlit sidebar.
    e.g. {"Python": 42, "JavaScript": 18, "Markdown": 5}
    """
    from collections import Counter
    lang_counts = Counter(c.metadata.get("language", "text") for c in chunks)
    return dict(lang_counts.most_common())