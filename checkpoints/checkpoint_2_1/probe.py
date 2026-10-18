import sys
from capstone_retriever import PaperRetriever

r = PaperRetriever(r".\ResearchPapers")

for term in sys.argv[1:]:
    print(f"\n=== {term} ===")
    for cid, txt, s in r._bm25_topk(term, 5):
        snippet = " ".join(txt.split())[:100]
        print(f"  {s:7.2f}  {cid}")
        print(f"           {snippet}")