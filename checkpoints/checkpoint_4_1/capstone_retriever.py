import os
import re

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from capstone_corpus import load_papers, chunk_papers

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\chroma_papers"        # NOT chroma_db — that holds the emails
CANDIDATE_POOL = 20
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

_STOPWORDS = {"a","an","the","and","or","in","on","at","to","of","by","with",
              "from","is","are","was","were","be","this","that","for","as","it"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def _normalize(scores, invert=False):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    normed = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in normed] if invert else normed


class PaperRetriever:
    def __init__(self, papers_dir: str, limit=None, chroma_dir=CHROMA_DIR):
        self._chunks = chunk_papers(load_papers(papers_dir, limit=limit))
        self._bm25 = BM25Okapi([tokenize(c.page_content) for c in self._chunks])

        if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
            print(f"Loading vector DB from {chroma_dir}/")
            self._db = Chroma(persist_directory=chroma_dir,
                              embedding_function=get_embeddings())
        else:
            print(f"Embedding {len(self._chunks)} chunks...")
            self._db = Chroma.from_documents(self._chunks, get_embeddings(),
                                             persist_directory=chroma_dir)
#class PaperRetriever:
#    def __init__(self, papers_dir: str):
#        self._chunks = chunk_papers(load_papers(papers_dir))
#        self._bm25 = BM25Okapi([tokenize(c.page_content) for c in self._chunks])
#
#        if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
#            print(f"Loading vector DB from {CHROMA_DIR}/")
#            self._db = Chroma(persist_directory=CHROMA_DIR,
#                              embedding_function=get_embeddings())
#        else:
#            print("Embedding chunks (first run, this takes a while)...")
#            self._db = Chroma.from_documents(self._chunks, get_embeddings(),
#                                             persist_directory=CHROMA_DIR)

    def _bm25_topk(self, query, k):
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i].metadata["chunk_id"],
                 self._chunks[i].page_content, scores[i]) for i in top]

    def _vector_topk(self, query, k):
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata["chunk_id"], d.page_content, s) for d, s in results]

    #def getTopK(self, query, k):
    def getTopK(self, query, k, max_per_source=2):
        bm_hits = self._bm25_topk(query, CANDIDATE_POOL)
        vec_hits = self._vector_topk(query, CANDIDATE_POOL)

        bm_norm = _normalize([s for _, _, s in bm_hits])
        vec_norm = _normalize([s for _, _, s in vec_hits], invert=True)

        fused, contents = {}, {}
        for (cid, content, _), s in zip(bm_hits, bm_norm):
            contents[cid] = content
            fused[cid] = fused.get(cid, 0.0) + WEIGHT_BM25 * s
        for (cid, content, _), s in zip(vec_hits, vec_norm):
            contents[cid] = content
            fused[cid] = fused.get(cid, 0.0) + WEIGHT_VECTOR * s

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        out, seen = [], {}
        for cid, score in ranked:
            src = cid.split("#")[0]
            if seen.get(src, 0) >= max_per_source:
                continue
            seen[src] = seen.get(src, 0) + 1
            out.append((cid, contents[cid], score))
            if len(out) == k:
                break
        return out
        #ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        #return [(cid, contents[cid], score) for cid, score in ranked]