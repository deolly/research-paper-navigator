"""Paper similarity with proceedings volumes excluded, using top-N neighbours."""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from pypdf import PdfReader

from capstone_retriever import get_embeddings, CHROMA_DIR

PAPERS = Path(r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers")

def is_proceedings(p1: str, n_pages: int) -> bool:
    low = p1.lower()
    m = ("lecture notes in computer science", "editorial board",
         "commenced publication", "springer", "proceedings of the")
    return sum(x in low for x in m) >= 2 or n_pages > 150

print("Identifying proceedings volumes...")
excluded = set()
for p in sorted(PAPERS.glob("*.pdf")):
    try:
        r = PdfReader(str(p))
        p1 = " ".join((r.pages[0].extract_text() or "").split())
        if is_proceedings(p1, len(r.pages)):
            excluded.add(p.name)
    except Exception:
        pass
print(f"  {len(excluded)} volumes to exclude\n")

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
data = db.get(include=["embeddings", "metadatas"])

by_paper = defaultdict(list)
for vec, m in zip(np.array(data["embeddings"]), data["metadatas"]):
    src = m["source"]
    if src not in excluded:
        by_paper[src].append(vec)

papers = sorted(by_paper)
mat = np.array([np.mean(by_paper[p], axis=0) for p in papers])
mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
print(f"{len(papers)} papers after exclusion\n")

sim = mat @ mat.T
np.fill_diagonal(sim, -1)

A = "DBLP_journals_pacmpl_BowersOWGTES23.pdf"
B = "DBLP_journals_corr_abs-2006-08381.pdf"
ia, ib = papers.index(A), papers.index(B)

order = np.argsort(-sim[ia])
rank = int(np.where(order == ib)[0][0]) + 1
print(f"Stitch <-> DreamCoder: cosine {sim[ia, ib]:.4f}, rank {rank}\n")

print("Stitch's 12 nearest papers:")
for j in order[:12]:
    mark = "  <-- DreamCoder" if j == ib else ""
    print(f"  {sim[ia, j]:.4f}  {papers[j]}{mark}")

print("\nEdge counts with top-N per paper (floor 0.75):")
for N in (5, 8, 10, 15):
    edges = set()
    for i in range(len(papers)):
        for j in np.argsort(-sim[i])[:N]:
            if sim[i, j] > 0.75:
                edges.add(tuple(sorted((i, int(j)))))
    connected = int(np.where(order == ib)[0][0]) < N
    print(f"  N={N:2}: {len(edges):5,} edges   avg degree {2*len(edges)/len(papers):.1f}"
          f"   Stitch-DreamCoder connected: {connected}")