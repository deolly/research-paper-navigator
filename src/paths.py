"""Single source of truth for where the corpus and the vector index live.

Before this module existed, the absolute path to the PDF folder was written
into four separate files. A reviewer cloning the repository had to find and
edit all four. Now they set one variable in `.env` and nothing else.

Resolution order for the corpus:
    1. CORPUS_DIR from the environment (or .env)
    2. <repo root>/ResearchPapers

Nothing here touches the filesystem at import time, so importing any module
is free and cannot fail on a missing corpus. Callers that actually need the
PDFs ask with `required=True` and get one clear message instead of a
stack trace several frames deep.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "ResearchPapers"
DEFAULT_CHROMA = REPO_ROOT / "chroma_papers"


def corpus_dir(required: bool = False) -> Path:
    """Folder holding the corpus PDFs."""
    raw = os.environ.get("CORPUS_DIR", "").strip().strip('"').strip("'")
    path = Path(raw).expanduser() if raw else DEFAULT_CORPUS

    if required and not path.is_dir():
        raise SystemExit(
            f"\nCorpus folder not found:\n    {path}\n\n"
            f"Set CORPUS_DIR in .env to the folder holding the PDFs, or put them at\n"
            f"    {DEFAULT_CORPUS}\n\n"
            f"The PDFs are not in this repository -- see README.md, section 'Corpus'.\n"
        )
    return path


def chroma_dir() -> Path:
    """Folder holding the persisted Chroma vector index.

    Anchored to the repository root, not the current working directory.
    Keyed off cwd, running the same code from a different folder silently
    re-embedded all 23,560 chunks -- about ten minutes and a real API bill.
    """
    raw = os.environ.get("CHROMA_DIR", "").strip().strip('"').strip("'")
    return Path(raw).expanduser() if raw else DEFAULT_CHROMA


def count_pdfs(path: Path | None = None) -> int:
    path = path or corpus_dir()
    return len(list(path.glob("*.pdf"))) if path.is_dir() else 0
