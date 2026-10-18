"""Capstone 2.1 — three-way retrieval comparison + grounded answers."""
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from capstone_retriever import PaperRetriever

load_dotenv()

PAPERS_DIR = r".\ResearchPapers"
TOP_K = 5
LOG = Path("checkpoint_2_1_retrieval.log")

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not "
    "contain the answer, say so rather than guessing."
)


def my_representative_queries():
    return [
        "What is storyboard programming and how does it synthesize data structure manipulations?",
        "How do these systems learn reusable libraries of program abstractions?",
        "How can a programmer leave gaps in their code for the computer to fill in automatically?",
        "How can a system automatically help students fix mistakes in their programming assignments?",
        "What is the current market capitalization of Nvidia?",
    ]


def log(text):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main():
    r = PaperRetriever(PAPERS_DIR)
    llm = ChatOpenAI(
        model="openai/gpt-5.4-mini",
        temperature=0.2,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    log(f"\n{'='*78}\nRUN {datetime.now().isoformat(timespec='seconds')}\n{'='*78}")

    for i, q in enumerate(my_representative_queries(), 1):
        block = [f"\n{'='*78}", f"QUERY {i}: {q}", "="*78]

        block.append("\n-- BM25 only --")
        for cid, _, s in r._bm25_topk(q, TOP_K):
            block.append(f"  {s:8.3f}  {cid}")

        block.append("\n-- Vector only --")
        for cid, _, s in r._vector_topk(q, TOP_K):
            block.append(f"  {s:8.3f}  {cid}")

        block.append("\n-- Hybrid (fused) --")
        hits = r.getTopK(q, TOP_K)
        for cid, _, s in hits:
            block.append(f"  {s:8.3f}  {cid}")

        context = "\n\n---\n\n".join(f"[{cid}]\n{txt}" for cid, txt, _ in hits)
        ans = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {q}"),
        ]).content

        block.append(f"\n-- ANSWER --\n{ans}")

        out = "\n".join(block)
        print(out)
        log(out)

    print(f"\nLogged to {LOG.resolve()}")


if __name__ == "__main__":
    main()