"""Check whether 2-hop similarity or author edges connect Stitch to DreamCoder."""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from pypdf import PdfReader

from capstone_retriever import get_embeddings, CHROMA_DIR

PAPERS = Path(r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers")
EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
BRACE = re.compile(r"\{([^}]+)\}@([\w.\-]+\.\w+)")

def is_proceedings(p1, n):
    low = p1.lower()
    m = ("lecture notes in computer science", "editorial board",
         "commenced publication", "springer", "proceedings of the")
    return sum(x in low for x in m) >= 2 or n > 150

print("Extracting emails and filtering volumes...")
excluded, emails = set(), {}
for p in sorted(PAPERS.glob("*.pdf")):
    try:
        r = PdfReader(str(p))
        p1 = " ".join((r.pages[0].extract_text() or "").split())
        if is_proceedings(p1, len(r.pages)):
            excluded.add(p.name)
            continue
        e = set(EMAIL.findall(p1))
        for names, dom in BRACE.findall(p1):
            for n in names.split(","):
                e.add(f"{n.strip()}@{dom}")
        emails[p.name] = e
    except Exception:
        pass

with_email = sum(1 for v in emails.values() if v)
print(f"  {len(excluded)} volumes excluded")
print(f"  {with_email}/{len(emails)} papers yielded at least one email\n")

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
data = db.get(include=["embeddings", "metadatas"])
by_paper = defaultdict(list)
for vec, m in zip(np.array(data["embeddings"]), data["metadatas"]):
    if m["source"] not in excluded:
        by_paper[m["source"]].append(vec)

papers = sorted(by_paper)
idx = {p: i for i, p in enumerate(papers)}
mat = np.array([np.mean(by_paper[p], axis=0) for p in papers])
mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
sim = mat @ mat.T
np.fill_diagonal(sim, -1)

A = "DBLP_journals_pacmpl_BowersOWGTES23.pdf"   # Stitch
B = "DBLP_journals_corr_abs-2006-08381.pdf"     # DreamCoder
C = "DBLP_conf_nips_EllisMSST18.pdf"            # Ellis 2018

for label, name in (("Stitch", A), ("DreamCoder", B), ("Ellis2018", C)):
    print(f"{label:<12} emails: {sorted(emails.get(name, []))}")

print(f"\nEllis2018 <-> DreamCoder cosine: {sim[idx[C], idx[B]]:.4f}")
order_c = np.argsort(-sim[idx[C]])
rank = int(np.where(order_c == idx[B])[0][0]) + 1
print(f"DreamCoder's rank among Ellis2018's neighbours: {rank}")

print(f"\nEllis2018's 10 nearest:")
for j in order_c[:10]:
    mark = "  <-- DreamCoder" if j == idx[B] else ""
    print(f"  {sim[idx[C], j]:.4f}  {papers[j]}{mark}")

# Two-hop reachability at several N
print("\n2-hop reachability Stitch -> DreamCoder (similar_to, top-N):")
for N in (5, 8, 10):
    nbr = {i: set(int(j) for j in np.argsort(-sim[i])[:N] if sim[i, j] > 0.75)
           for i in range(len(papers))}
    hop1 = nbr[idx[A]]
    hop2 = set().union(*(nbr[h] for h in hop1)) if hop1 else set()
    reach = idx[B] in (hop1 | hop2)
    print(f"  N={N:2}: 1-hop={len(hop1)}  2-hop total={len(hop1 | hop2)}  reaches DreamCoder: {reach}")

# Author-edge route
sa = emails.get(A, set()); sb = emails.get(B, set()); sc = emails.get(C, set())
print(f"\nsame_author edges:")
print(f"  Stitch & DreamCoder share: {sorted(sa & sb)}")
print(f"  Ellis2018 & DreamCoder share: {sorted(sc & sb)}")
print(f"  Stitch & Ellis2018 share: {sorted(sa & sc)}")