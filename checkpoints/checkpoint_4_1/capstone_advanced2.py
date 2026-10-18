"""capstone_advanced.py — advanced retrieval: query decomposition + graph traversal.

Pipeline
    query
      -> decompose into 2-4 sub-queries (one LLM call)
      -> hybrid-retrieve for each sub-query, sum fused scores per chunk
      -> take the top N_PRIMARY chunks as PRIMARY
      -> walk the corpus graph from those chunks' papers (1 and 2 hops)
      -> pick N_CONTEXT neighbour papers, one best-matching chunk each, as CONTEXT
      -> prompt the LLM with PRIMARY and CONTEXT labelled separately

Document count is held at N_PRIMARY + N_CONTEXT == baseline TOP_K, so a
baseline-vs-advanced comparison is not confounded by context size.

duplicate_of edges are used to SUPPRESS rather than expand: a paper that
duplicates one already selected adds no information, and in Checkpoint 2.1 a
preprint/conference pair consumed two of five slots on the E4 query.

Modes
    baseline        single-pass hybrid retrieval (the Checkpoint 2.1 system)
    decompose       decomposition only, no graph
    graph           graph expansion only, no decomposition
    both            decomposition + graph expansion

Usage
    python capstone_advanced.py --mode both --query "..."
    python capstone_advanced.py --compare        # all four modes on the test queries
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from capstone_graph import load_graph
from capstone_retriever import PaperRetriever, tokenize

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.0              # deterministic, per the Checkpoint 3.1 finding
PAPERS_DIR = r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers"

TOP_K = 5                      # total chunks sent to the LLM, both modes
N_PRIMARY = 3                  # of which, retrieved directly
N_CONTEXT = 2                  # of which, reached through the graph
SUB_QUERY_POOL = 8             # chunks pulled per sub-query before merging
TWO_HOP_DISCOUNT = 0.4         # weight applied to 2-hop neighbours

LOG_PATH = Path("checkpoint_4_1_advanced.log")

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system "
    "over a corpus of computer science research papers. Break the user's "
    "question into 2-4 focused sub-queries that together cover everything "
    "needed to answer it. Each sub-query should target a distinct aspect, and "
    "should use the technical vocabulary a research paper would use. "
    'Return ONLY a JSON array of strings, e.g. ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant answering questions about a corpus of computer "
    "science research papers. Answer using ONLY the provided documents, and quote "
    "from them where you can. PRIMARY DOCUMENTS matched the question directly. "
    "RELATED CONTEXT documents were surfaced through citation and similarity "
    "links from the primary documents; use them when they add relevant "
    "information, especially when the question asks you to compare or contrast "
    "across papers. If the documents do not contain the answer, say so rather "
    "than guessing."
)


def make_llm() -> ChatOpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — check your .env file.")
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=key, base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 74}\n")


# ─── Lazily built, shared across calls ───────────────────────────────
_retriever: PaperRetriever | None = None
_graph: nx.Graph | None = None
_meta: dict | None = None
_chunks_by_source: dict[str, list[int]] | None = None


def get_retriever() -> PaperRetriever:
    global _retriever, _chunks_by_source
    if _retriever is None:
        _retriever = PaperRetriever(PAPERS_DIR)
        _chunks_by_source = defaultdict(list)
        for i, c in enumerate(_retriever._chunks):
            _chunks_by_source[c.metadata["source"]].append(i)
    return _retriever


def get_graph() -> tuple[nx.Graph, dict]:
    global _graph, _meta
    if _graph is None:
        _graph, _meta = load_graph()
    return _graph, _meta


# ─── Step 1: decomposition ───────────────────────────────────────────
def decompose_query(llm: ChatOpenAI, query: str) -> tuple[list[str], bool]:
    """Return (sub_queries, succeeded). On failure the original query is returned,
    flagged, so a silent fallback is not mistaken for the technique not helping."""
    raw = llm.invoke([
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=query),
    ]).content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if "[" in raw:
            raw = raw[raw.find("["):raw.rfind("]") + 1]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
            return parsed, True
    except (json.JSONDecodeError, ValueError):
        pass
    return [query], False


# ─── Step 2: multi-query retrieval ───────────────────────────────────
def merged_retrieve(queries: list[str], k: int) -> list[tuple[str, str, float]]:
    """Retrieve for each query and sum fused scores per chunk.
    A chunk relevant to several sub-queries rises above one relevant to a single one."""
    r = get_retriever()
    totals: dict[str, float] = defaultdict(float)
    text: dict[str, str] = {}

    for q in queries:
        for cid, content, score in r.getTopK(q, SUB_QUERY_POOL):
            totals[cid] += score
            text[cid] = content

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(cid, text[cid], s) for cid, s in ranked]


# ─── Step 3: graph expansion ─────────────────────────────────────────
def expand_via_graph(primary: list[tuple[str, str, float]], n_context: int
                     ) -> list[tuple[str, str, float, str]]:
    """From the papers of the primary chunks, walk the graph and pick n_context
    neighbour papers. Returns (chunk_id, content, score, provenance)."""
    G, _ = get_graph()
    r = get_retriever()

    seed_papers = {cid.split("#")[0] for cid, _, _ in primary}
    seed_weight = defaultdict(float)
    for cid, _, score in primary:
        seed_weight[cid.split("#")[0]] += score

    # Papers to suppress: the seeds themselves, plus anything duplicating a seed
    suppressed = set(seed_papers)
    for p in seed_papers:
        if p in G:
            for nb in G.neighbors(p):
                if G.edges[p, nb]["edge_type"] == "duplicate_of":
                    suppressed.add(nb)

    # 1-hop
    candidates: dict[str, tuple[float, str]] = {}
    for p in seed_papers:
        if p not in G:
            continue
        for nb in G.neighbors(p):
            et = G.edges[p, nb]["edge_type"]
            if et == "duplicate_of" or nb in suppressed:
                continue
            w = G.edges[p, nb].get("weight", 1.0)
            score = seed_weight[p] * w
            if nb not in candidates or score > candidates[nb][0]:
                candidates[nb] = (score, f"{et} from {p.split('.')[0][:28]}")

    # 2-hop, discounted
    for mid in list(candidates):
        if mid not in G:
            continue
        for nb in G.neighbors(mid):
            et = G.edges[mid, nb]["edge_type"]
            if et == "duplicate_of" or nb in suppressed or nb in candidates:
                continue
            w = G.edges[mid, nb].get("weight", 1.0)
            score = candidates[mid][0] * w * TWO_HOP_DISCOUNT
            candidates[nb] = (score, f"2-hop {et} via {mid.split('.')[0][:24]}")

    # Round-robin across seed papers in rank order, so every primary paper gets a
    # chance to contribute context before the highest-scoring seed takes both slots.
    ordered_seeds = []
    for cid, _, _ in primary:
        p = cid.split("#")[0]
        if p not in ordered_seeds:
            ordered_seeds.append(p)

    chosen, seen = [], set()
    for p in ordered_seeds:
        if len(chosen) >= n_context:
            break
        tag = p.split(".")[0][:28]
        best = None
        for nb, (score, prov) in candidates.items():
            if nb in seen or tag not in prov:
                continue
            if best is None or score > best[1][0]:
                best = (nb, (score, prov))
        if best:
            chosen.append(best)
            seen.add(best[0])

    # Top up from the global ranking if the seeds did not yield enough
    for nb, v in sorted(candidates.items(), key=lambda kv: -kv[1][0]):
        if len(chosen) >= n_context:
            break
        if nb not in seen:
            chosen.append((nb, v))
            seen.add(nb)

    # One best-matching chunk per chosen paper, scored against the primary text
    out = []
    probe = " ".join(c for _, c, _ in primary)[:2000]
    bm_scores = r._bm25.get_scores(tokenize(probe))
    for paper, (gscore, prov) in chosen:
        idxs = _chunks_by_source.get(paper, [])
        if not idxs:
            continue
        best = max(idxs, key=lambda i: bm_scores[i])
        doc = r._chunks[best]
        out.append((doc.metadata["chunk_id"], doc.page_content, gscore, prov))
    return out


# ─── Step 4: assemble context and answer ─────────────────────────────
def build_context(primary, context) -> str:
    parts = ["PRIMARY DOCUMENTS (matched the question directly):"]
    for cid, content, _ in primary:
        parts.append(f"[{cid}]\n{content}")
    if context:
        parts.append("\nRELATED CONTEXT (reached through citation/similarity links):")
        for cid, content, _, prov in context:
            parts.append(f"[{cid}]  ({prov})\n{content}")
    return "\n\n---\n\n".join(parts)


def retrieve(mode: str, query: str, llm: ChatOpenAI, verbose: bool = True):
    """Return (primary, context, sub_queries). Total chunks == TOP_K in every mode."""
    sub_queries, ok = [query], True

    if mode in ("decompose", "both"):
        sub_queries, ok = decompose_query(llm, query)
        if verbose:
            if ok:
                print(f"  decomposed into {len(sub_queries)}:")
                for sq in sub_queries:
                    print(f"      - {sq}")
            else:
                print("  !! decomposition failed to parse — fell back to raw query")

    use_graph = mode in ("graph", "both")
    n_primary = N_PRIMARY if use_graph else TOP_K

    primary = merged_retrieve(sub_queries, n_primary)
    context = expand_via_graph(primary, N_CONTEXT) if use_graph else []
    return primary, context, sub_queries


def answer(llm: ChatOpenAI, query: str, primary, context) -> str:
    return llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"{build_context(primary, context)}\n\nQuestion: {query}"),
    ]).content


# ─── Reporting ───────────────────────────────────────────────────────
def show(mode: str, query: str, primary, context, ans: str) -> None:
    papers = {cid.split("#")[0] for cid, _, _ in primary}
    papers |= {cid.split("#")[0] for cid, _, _, _ in context}

    print(f"\n  PRIMARY ({len(primary)}):")
    for cid, _, s in primary:
        print(f"      {s:6.3f}  {cid}")
    if context:
        print(f"  CONTEXT ({len(context)}):")
        for cid, _, s, prov in context:
            print(f"      {s:6.3f}  {cid}")
            print(f"              via {prov}")
    print(f"  total chunks: {len(primary) + len(context)}   distinct papers: {len(papers)}")
    print(f"\n  ANSWER:\n{ans}\n")

    log(f"MODE={mode}  QUERY={query}",
        f"primary={[c for c,_,_ in primary]}\n"
        f"context={[(c, p) for c,_,_,p in context]}\n"
        f"distinct_papers={len(papers)}\nanswer={ans}")


TEST_QUERIES = [
    "How can a programmer leave gaps in their code for the computer to fill in "
    "automatically?",
    "Which different approaches to learning reusable libraries of program "
    "abstractions appear in this corpus, and how do they differ from each other?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["baseline", "decompose", "graph", "both"])
    ap.add_argument("--query", help="a single query to run")
    ap.add_argument("--compare", action="store_true",
                    help="run all four modes on both test queries")
    args = ap.parse_args()

    llm = make_llm()

    queries = [args.query] if args.query else TEST_QUERIES
    modes = ["baseline", "decompose", "graph", "both"] if args.compare else [args.mode]

    for q in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)
        for m in modes:
            print(f"\n--- MODE: {m} ---")
            primary, context, _ = retrieve(m, q, llm)
            ans = answer(llm, q, primary, context)
            show(m, q, primary, context, ans)

    print(f"\nLogged to {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
