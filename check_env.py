"""Preflight check: confirm this machine can run the Research Paper Navigator.

Run this before anything else. It makes no API calls, costs nothing, and
takes about a second:

    python check_env.py

It verifies four things in order, because each depends on the one before:
dependencies, then the system's own modules, then configuration, then data.
A failure at any level explains what to do instead of leaving a traceback.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"
problems: list[str] = []


def check(label: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    print(f"{PASS if ok else (FAIL if fatal else WARN)}  {label}"
          + (f"  --  {detail}" if detail else ""))
    if not ok and fatal:
        problems.append(label)
    return ok


print("\n" + "=" * 72)
print("  Research Paper Navigator  --  environment check")
print("=" * 72)

# ---------------------------------------------------------------- 1. runtime
print("\n[1/4] Python runtime")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 10),
      "3.10 or newer required (the code uses `X | None` annotations)")

# ----------------------------------------------------------- 2. dependencies
print("\n[2/4] Third-party packages")
DEPS = [
    ("dotenv", "python-dotenv"),
    ("langchain_core", "langchain-core"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("chromadb", "chromadb"),
    ("rank_bm25", "rank-bm25"),
    ("networkx", "networkx"),
    ("numpy", "numpy"),
    ("pypdf", "pypdf"),
]
for mod, pkg in DEPS:
    try:
        m = importlib.import_module(mod)
        check(pkg, True, getattr(m, "__version__", ""))
    except Exception as e:
        check(pkg, False, f"{type(e).__name__}: {e}  ->  pip install -r requirements.txt")

# --------------------------------------------------------------- 3. modules
print("\n[3/4] System modules (src/)")
MODULES = [
    "paths",
    "capstone_corpus",
    "capstone_retriever",
    "capstone_graph",
    "capstone_advanced",
    "capstone_agent",
    "capstone_security_cost",
    "capstone_production",
    "capstone_eval_3_1",
    "capstone_eval_4_1",
]
for mod in MODULES:
    try:
        importlib.import_module(mod)
        check(f"{mod}.py", True)
    except Exception as e:
        check(f"{mod}.py", False, f"{type(e).__name__}: {e}")

# --------------------------------------------------- 4. configuration + data
print("\n[4/4] Configuration and corpus")
key = os.environ.get("OPENROUTER_API_KEY", "")
check("OPENROUTER_API_KEY", bool(key),
      f"set, {len(key)} chars" if key else
      "missing  ->  copy .env.example to .env and add your key")

try:
    from paths import corpus_dir, chroma_dir, count_pdfs
    cdir = corpus_dir()
    n = count_pdfs(cdir)
    check(f"corpus at {cdir}", n > 0,
          f"{n} PDFs" if n else "no PDFs found  ->  set CORPUS_DIR in .env")
    if n and n != 153:
        check("corpus size", True, f"{n} PDFs (results in evidence/ "
              f"were measured on 153)", fatal=False)
    xdir = chroma_dir()
    built = xdir.is_dir() and any(xdir.iterdir())
    check(f"vector index at {xdir}", built,
          "prebuilt, ready" if built else
          "absent -- first run will embed ~23,560 chunks (~10 min, costs money)",
          fatal=False)
except Exception as e:
    check("corpus resolution", False, f"{type(e).__name__}: {e}")

# ----------------------------------------------------------------- verdict
print("\n" + "=" * 72)
if problems:
    print(f"  NOT READY -- {len(problems)} problem(s): " + ", ".join(problems))
    print("=" * 72 + "\n")
    sys.exit(1)
print("  READY -- every check passed.")
print("=" * 72 + "\n")
