"""
pipeline/loader.py
Clones a GitHub repo (shallow) and loads all readable code files
into LangChain Document objects with clean metadata.
"""

import os
import tempfile
import shutil
from pathlib import Path
from git import Repo, GitCommandError
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

# Folders to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", "target", "vendor",
    "bower_components", ".idea", ".vscode",
}

# wanted file
CODE_EXTENSIONS = {
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    # Java
    ".java",
    # Go
    ".go",
    # Rust
    ".rs",
    # C / C++
    ".c", ".cpp", ".cc", ".h", ".hpp",
    # Web
    ".html", ".css", ".scss",
    # Config / Docs
    ".md", ".yaml", ".yml", ".toml", ".json", ".env.example",
    # Shell
    ".sh", ".bash",
    # SQL
    ".sql",
}

MAX_FILE_SIZE_BYTES = 500_000  # skip files larger than 500 KB
MAX_TOTAL_FILES = 300  


def clone_repo(repo_url: str) -> str:
    """
    Clone a GitHub repo into a temp directory.
    Returns the path to the cloned repo.
    Uses shallow clone (depth=1) to be fast.
    """
    tmp_dir = tempfile.mkdtemp(prefix="codebuddy_")
    try:
        print(f"[Loader] Cloning {repo_url} ...")
        Repo.clone_from(
            repo_url,
            tmp_dir,
            depth=1,  # shallow — only latest commit
            single_branch=True,
        )
        print(f"[Loader] Cloned to {tmp_dir}")
        return tmp_dir
    except GitCommandError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(
            f"Could not clone repo.\n"
            f"Check: (1) URL is correct, (2) repo is public, (3) git is installed.\n"
            f"Error: {e}"
        )


def load_repo(repo_url: str) -> tuple[list[Document], str]:
    """
    Main entry point.
    Clones the repo and returns (list[Document], tmp_dir_path).
    Caller is responsible for cleanup of tmp_dir.
    """
    tmp_dir = clone_repo(repo_url)
    # clone repo
    docs = _walk_and_load(tmp_dir)
    # load evry file and clean return clean list[Document]
    return docs, tmp_dir


def _walk_and_load(root: str) -> list[Document]:
    """
    Walk the cloned directory tree and load every code file
    into a LangChain Document with rich metadata.
    """
    docs = []
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk skips them
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            if file_count >= MAX_TOTAL_FILES:
                print(f"[Loader] Hit file cap ({MAX_TOTAL_FILES}). Stopping early.")
                return docs

            ext = Path(fname).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            fpath = os.path.join(dirpath, fname)

            # Skip files that are too large
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if size > MAX_FILE_SIZE_BYTES:
                print(f"[Loader] Skipping large file: {fname} ({size // 1024} KB)")
                continue

            # Load file content
            try:
                loader = TextLoader(fpath, encoding="utf-8", autodetect_encoding=True)
                file_docs = loader.load()
            except Exception as ex:
                print(f"[Loader] Could not read {fname}: {ex}")
                continue

            # Enrich metadata for every document from this file
            rel_path = os.path.relpath(fpath, root)
            for doc in file_docs:
                doc.metadata["source"] = rel_path  # e.g. "src/utils/helpers.py"
                doc.metadata["filename"] = fname  # e.g. "helpers.py"
                doc.metadata["extension"] = ext  # e.g. ".py"
                doc.metadata["size_kb"] = round(size / 1024, 1)
                doc.metadata["directory"] = os.path.relpath(dirpath, root)

            docs.extend(file_docs)
            file_count += 1

    print(f"[Loader] Loaded {len(docs)} files ({file_count} unique paths)")
    return docs
