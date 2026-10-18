"""Test paper-level embedding similarity as a shares_topic replacement."""
import numpy as np
from collections import defaultdict

from capstone_retriever import get_embeddings, CHROMA_DIR
from langchain_chroma import Chroma

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

print("Pulling stored embeddings from Chroma...")
data = db.get(include=["embeddings", "metadatas"])
embs = np.array(data["embeddings"])
metas = data["metadatas"]
print(f"  {len(embs)} chunk embeddings")

# Average chunk vectors per source paper
by_paper = defaultdict(list)
for vec, m in zip(embs, metas):
    by_paper[m["source"]].append(vec)

papers = sorted(by_paper)
mat = np.array([np.mean(by_paper[p], axis=0) for p in papers])
mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
print(f"  {len(papers)} paper-level vectors\n")

sim = mat @ mat.T
np.fill_diagonal(sim, -1)

# Edge density at several thresholds
print("similar_to edge counts by cosine threshold:")
for t in (0.70, 0.75, 0.80, 0.85, 0.90):
    n = int((sim > t).sum() / 2)
    print(f"  > {t:.2f}: {n:6,} edges   (avg degree {2*n/len(papers):.1f})")

# The E4 pair
A = "DBLP_journals_pacmpl_BowersOWGTES23.pdf"   # Stitch
B = "DBLP_journals_corr_abs-2006-08381.pdf"     # DreamCoder
if A in papers and B in papers:
    ia, ib = papers.index(A), papers.index(B)
    print(f"\nStitch <-> DreamCoder cosine: {sim[ia, ib]:.4f}")
    print(f"\nStitch's 8 nearest papers:")
    for j in np.argsort(-sim[ia])[:8]:
        print(f"  {sim[ia, j]:.4f}  {papers[j]}")
else:
    print(f"\nMissing: Stitch={A in papers} DreamCoder={B in papers}")