"""capstone_graph.py — build a paper-level relationship graph over the corpus.

Node types
    paper   one per PDF, after excluding proceedings volumes

Edge types
    similar_to    paper-level mean embedding, top-N neighbours above a cosine floor
    cites         space-insensitive title match inside the reference text
    duplicate_of  near-identical titles (preprint / conference pairs)

Design decisions and the evidence behind them:
  - 15 proceedings volumes are excluded. They run to 939 pages, average to a
    centroid vector near the middle of the field, and act as hubs connected to
    everything. They were also the source of the irrelevant BM25 hits observed
    in Checkpoint 2.1 on the E3 query.
  - same_author was tested and dropped. Author emails were extractable from only
    106 of 138 papers, and the two papers E4 depends on (Stitch, DreamCoder)
    share none — DreamCoder is a preprint that lists no addresses at all.
  - shares_topic (keyword bigram overlap) was tested and dropped. Stitch and
    DreamCoder shared zero keywords at any threshold, because bigram frequency
    captures what makes a paper distinctive rather than what it is about.
  - similar_to uses top-N per paper rather than a global cosine threshold, so
    every paper gets neighbours regardless of where it sits in similarity space,
    and hub degree stays bounded.

Usage
    python capstone_graph.py           build (or rebuild) and print a summary
    python capstone_graph.py --force   ignore the cache and re-parse every PDF

    from capstone_graph import load_graph
    G, papers = load_graph()
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from langchain_chroma import Chroma
from pypdf import PdfReader

from capstone_retriever import CHROMA_DIR, get_embeddings
from paths import corpus_dir

# ─── Configuration ───────────────────────────────────────────────────
PAPERS_DIR = corpus_dir()
CACHE_DIR = Path("graph_cache")
META_CACHE = CACHE_DIR / "paper_metadata.json"
GRAPH_CACHE = CACHE_DIR / "corpus_graph.pkl"

SIMILAR_TOP_N = 5        # neighbours per paper; probe showed 2-hop reaches
SIMILAR_FLOOR = 0.75     # DreamCoder from Stitch even at N=5
DUPLICATE_FLOOR = 0.97   # title similarity for preprint/conference pairs
TITLE_PROBE_LEN = 45     # chars of a squashed title used for citation matching
REF_START_FRAC = 0.15    # references begin ~25% in; search from 15% to be safe

EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
BRACE = re.compile(r"\{([^}]+)\}@([\w.\-]+\.\w+)")


# ─── Text helpers ────────────────────────────────────────────────────
def squash(s: str) -> str:
    """Lowercase, drop everything non-alphanumeric.

    PDF extraction mangles small-caps titles ('END-TO-ENDSPATIALCONCEPTLEARN- ING'),
    so spacing cannot be trusted. Removing it entirely makes matching robust.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_proceedings(page1: str, n_pages: int) -> bool:
    low = page1.lower()
    markers = ("lecture notes in computer science", "editorial board",
               "commenced publication", "springer", "proceedings of the")
    return sum(m in low for m in markers) >= 2 or n_pages > 150


def guess_title(page1: str) -> str:
    """Text preceding the abstract marker — a serviceable title guess."""
    t = page1
    for marker in ("Abstract", "ABSTRACT", "abstract"):
        i = t.find(marker)
        if 20 < i < 500:
            t = t[:i]
            break
    return t[:200].strip()


def extract_emails(page1: str) -> list[str]:
    """Kept for reporting coverage, not used for edges — see module docstring."""
    found = set(EMAIL.findall(page1))
    for names, domain in BRACE.findall(page1):
        for n in names.split(","):
            found.add(f"{n.strip()}@{domain}")
    return sorted(found)


# ─── Stage 1: parse PDFs once, cache the result ──────────────────────
def build_metadata(force: bool = False) -> dict:
    if META_CACHE.exists() and not force:
        print(f"Loading cached metadata from {META_CACHE}")
        return json.loads(META_CACHE.read_text(encoding="utf-8"))

    CACHE_DIR.mkdir(exist_ok=True)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Parsing {len(pdfs)} PDFs (one-time, then cached)...")

    meta, excluded = {}, []
    for i, path in enumerate(pdfs, 1):
        if i % 25 == 0:
            print(f"  {i}/{len(pdfs)}")
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        page1 = " ".join((reader.pages[0].extract_text() or "").split())
        if is_proceedings(page1, len(reader.pages)):
            excluded.append(path.name)
            continue

        title = guess_title(page1)
        title_sq = squash(title)
        if len(title_sq) < 30:
            excluded.append(path.name)
            continue

        start = int(len(reader.pages) * REF_START_FRAC)
        ref_text = squash(" ".join(
            (pg.extract_text() or "") for pg in reader.pages[start:]
        ))

        meta[path.name] = {
            "title": title,
            "title_sq": title_sq,
            "n_pages": len(reader.pages),
            "emails": extract_emails(page1),
            "ref_sq": ref_text,
        }

    print(f"  kept {len(meta)} papers, excluded {len(excluded)}")
    META_CACHE.write_text(json.dumps({"papers": meta, "excluded": excluded}),
                          encoding="utf-8")
    return {"papers": meta, "excluded": excluded}


# ─── Stage 2: paper-level vectors from the existing Chroma store ─────
def paper_vectors(keep: set[str]) -> tuple[list[str], np.ndarray]:
    print("Loading chunk embeddings from Chroma...")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
    data = db.get(include=["embeddings", "metadatas"])

    by_paper = defaultdict(list)
    for vec, m in zip(np.array(data["embeddings"]), data["metadatas"]):
        src = m.get("source")
        if src in keep:
            by_paper[src].append(vec)

    names = sorted(by_paper)
    mat = np.array([np.mean(by_paper[n], axis=0) for n in names])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    print(f"  {len(names)} paper vectors from {len(data['metadatas'])} chunks")
    return names, mat


# ─── Stage 3: assemble the graph ─────────────────────────────────────
def build_graph(force: bool = False) -> tuple[nx.Graph, dict]:
    if GRAPH_CACHE.exists() and not force:
        print(f"Loading cached graph from {GRAPH_CACHE}")
        with GRAPH_CACHE.open("rb") as fh:
            return pickle.load(fh)

    md = build_metadata(force)
    papers_meta = md["papers"]
    names, mat = paper_vectors(set(papers_meta))
    idx = {n: i for i, n in enumerate(names)}

    G = nx.Graph()
    for n in names:
        G.add_node(n, node_type="paper",
                   title=papers_meta[n]["title"],
                   n_pages=papers_meta[n]["n_pages"])

    # similar_to — top-N neighbours per paper above the floor
    sim = mat @ mat.T
    np.fill_diagonal(sim, -1.0)
    n_sim = 0
    for i, name in enumerate(names):
        for j in np.argsort(-sim[i])[:SIMILAR_TOP_N]:
            if sim[i, j] > SIMILAR_FLOOR:
                if not G.has_edge(name, names[j]):
                    G.add_edge(name, names[j], edge_type="similar_to",
                               weight=float(sim[i, j]))
                    n_sim += 1

    # duplicate_of — near-identical titles (preprint / conference pairs)
    n_dup = 0
    for a in range(len(names)):
        ta = papers_meta[names[a]]["title_sq"][:60]
        for b in range(a + 1, len(names)):
            tb = papers_meta[names[b]]["title_sq"][:60]
            if len(ta) > 30 and len(tb) > 30 and (ta in tb or tb in ta):
                G.add_edge(names[a], names[b], edge_type="duplicate_of", weight=1.0)
                n_dup += 1

    # cites — squashed title of B appears in A's reference text
    n_cite = 0
    for a in names:
        ref = papers_meta[a]["ref_sq"]
        if not ref:
            continue
        for b in names:
            if a == b:
                continue
            probe = papers_meta[b]["title_sq"][:TITLE_PROBE_LEN]
            if len(probe) >= TITLE_PROBE_LEN and probe in ref:
                if not G.has_edge(a, b):
                    G.add_edge(a, b, edge_type="cites", weight=1.0)
                    n_cite += 1

    print(f"\nEdges added: similar_to={n_sim}  duplicate_of={n_dup}  cites={n_cite}")

    CACHE_DIR.mkdir(exist_ok=True)
    with GRAPH_CACHE.open("wb") as fh:
        pickle.dump((G, papers_meta), fh)
    print(f"Cached to {GRAPH_CACHE}")
    return G, papers_meta


def load_graph(force: bool = False):
    return build_graph(force)


# ─── Summary and validation ──────────────────────────────────────────
def summarise(G: nx.Graph, meta: dict) -> None:
    by_type = defaultdict(int)
    for _, _, d in G.edges(data=True):
        by_type[d["edge_type"]] += 1

    print("\n" + "=" * 74)
    print("  GRAPH SUMMARY")
    print("=" * 74)
    print(f"  nodes (papers): {G.number_of_nodes()}")
    print(f"  edges total:    {G.number_of_edges()}")
    for t, n in sorted(by_type.items()):
        print(f"      {t:<14} {n}")
    degrees = [d for _, d in G.degree()]
    print(f"  degree: mean {np.mean(degrees):.1f}  median {np.median(degrees):.0f}  "
          f"max {max(degrees)}")
    isolated = [n for n, d in G.degree() if d == 0]
    print(f"  isolated papers: {len(isolated)}")

    hubs = sorted(G.degree(), key=lambda kv: -kv[1])[:5]
    print("\n  Highest-degree papers (check these are not hubs by accident):")
    for n, d in hubs:
        print(f"      {d:3}  {n[:60]}")

    # The E4 path this graph was designed to support
    A = "DBLP_journals_pacmpl_BowersOWGTES23.pdf"   # Stitch
    B = "DBLP_journals_corr_abs-2006-08381.pdf"     # DreamCoder
    print("\n  E4 validation — Stitch to DreamCoder:")
    if A in G and B in G:
        try:
            path = nx.shortest_path(G, A, B)
            print(f"      reachable in {len(path) - 1} hop(s):")
            for step in path:
                print(f"          {step}")
        except nx.NetworkXNoPath:
            print("      NOT REACHABLE")
    else:
        print(f"      missing node: Stitch={A in G} DreamCoder={B in G}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore caches, re-parse PDFs")
    args = ap.parse_args()

    G, meta = build_graph(force=args.force)
    summarise(G, meta)