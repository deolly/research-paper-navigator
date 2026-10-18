r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis.

ADAPTED for the Research Paper Navigator capstone scenario.

Replaces the starter's six-document demonstration corpus with the hybrid
retriever built in Checkpoint 2.1 (BM25 + dense vectors over 23,560 chunks
extracted from 153 research-paper PDFs).

Run:  python capstone_checkpoint_3_1_evaluation.py

Requires, in the same folder:
    capstone_corpus.py      - PDF loading and chunking
    capstone_retriever.py   - PaperRetriever (BM25 + vector + fusion)
    chroma_papers/          - the persisted vector store
    ResearchPapers/         - the PDF corpus
    .env                    - OPENROUTER_API_KEY=sk-or-...
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from capstone_retriever import PaperRetriever

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANSWER_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4-mini"   # see cross_model_check() — same model is a known pitfall
TEMPERATURE = 0
TOP_K = 5                              # matches Checkpoint 2.1 so results are comparable
PAPERS_DIR = r".\ResearchPapers"
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

SCENARIO = "research_papers"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# ─── Infrastructure ──────────────────────────────────────────────────
def check_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in a .env file next to this "
            "script and rerun."
        )
    return key


def make_llm(model: str = ANSWER_MODEL) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# ─── Retrieval and answering, wired to the Checkpoint 2.1 system ─────
_retriever: PaperRetriever | None = None
_chunk_text: dict[str, str] = {}


def get_retriever() -> PaperRetriever:
    """Built once and reused — rebuilding the BM25 index over 23,560 chunks is slow."""
    global _retriever
    if _retriever is None:
        _retriever = PaperRetriever(PAPERS_DIR)
    return _retriever


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Hybrid retrieval from Checkpoint 2.1. Returns (chunk_id, fused_score)."""
    hits = get_retriever().getTopK(query, k)
    for cid, content, _ in hits:
        _chunk_text[cid] = content
    return [(cid, score) for cid, _, score in hits]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[{cid}]\n{_chunk_text[cid]}" for cid in doc_ids if cid in _chunk_text
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# ─── The metric ──────────────────────────────────────────────────────
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    """LLM-judge returning pass/fail against grading notes.

    NOTE: the starter's parser was `"pass" if "pass" in verdict else "fail"`, which
    scores the reply "the answer does not pass" as a PASS. This version anchors on
    the first token and treats anything ambiguous as a fail, then records the raw
    reply so parser behaviour can be audited. See parser_robustness_check().
    """
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\n"
                    f"Verdict (pass/fail):"
        ),
    ]
    raw = llm.invoke(messages).content.strip().lower()
    first = raw.split()[0].strip(".,:;'\"") if raw.split() else ""
    if first == "pass":
        return "pass"
    if first == "fail":
        return "fail"
    return "fail"   # ambiguous replies fail closed


def judge_raw(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    """Same call, but returns the judge's unparsed reply — used to audit the parser."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\n"
                    f"Verdict (pass/fail):"
        ),
    ]
    return llm.invoke(messages).content.strip()


# ─── The evaluation set ──────────────────────────────────────────────
def my_eval_set() -> list[dict]:
    """Five evaluation items, seeded from the Checkpoint 2.1 retrieval diagnosis.

    Two are expected to pass, two are expected to fail on known weaknesses, and one
    is an out-of-corpus control where the correct behaviour is refusal.
    """
    return [
        {
            "id": "E1-rare-term",
            "expectation": "pass",
            "why": "Rare coined term. In 2.1, BM25 scored 36.6 and the vector "
                   "retriever agreed at distance 0.512 — both retrievers converged.",
            "question": "What is storyboard programming and how does it synthesize "
                        "data structure manipulations?",
            "grading_notes": (
                "The answer must convey that storyboard programming synthesizes low-level "
                "imperative data-structure code from a high-level, example-based or "
                "diagram-like specification supplied by the programmer, rather than from "
                "a complete program. Exact terminology need not match."
            ),
        },
        {
            "id": "E2-accessible-paraphrase",
            "expectation": "pass",
            "why": "Paraphrased query whose everyday vocabulary happens to match the "
                   "source paper's own language. Passed retrieval in 2.1 at BM25 29.6.",
            "question": "How can a system automatically help students fix mistakes in "
                        "their programming assignments?",
            "grading_notes": (
                "The answer must describe using an instructor's reference implementation "
                "together with an error model of likely student mistakes to compute "
                "minimal corrections to the student's program."
            ),
        },
        {
            "id": "E3-vocabulary-gap",
            "expectation": "fail",
            "why": "SEEDED FAILURE. In 2.1 all three retrievers missed the four relevant "
                   "sketching papers. Probes showed the corpus says 'holes' and 'sketch' "
                   "where the query says 'gaps' and 'fill in'.",
            "question": "How can a programmer leave gaps in their code for the computer "
                        "to fill in automatically?",
            "grading_notes": (
                "The answer must explain that the programmer writes a partial program — "
                "a sketch — containing holes, and that a synthesizer automatically fills "
                "those holes so the program satisfies a specification."
            ),
        },
        {
            "id": "E4-multi-document",
            "expectation": "fail",
            "why": "SEEDED FAILURE. In 2.1 this query returned only two distinct papers "
                   "across five slots, one of them a duplicate pair; DreamCoder never "
                   "surfaced.",
            "question": "Which different approaches to learning reusable libraries of "
                        "program abstractions appear in this corpus, and how do they "
                        "differ from each other?",
            "grading_notes": (
                "The answer must name at least two systems by name (e.g. Stitch, DreamCoder) and "
                "describe a difference between their approaches. An answer drawn from a "
                "single source, however detailed, does not satisfy this."
            ),
        },
        {
            "id": "E5-out-of-corpus",
            "expectation": "pass",
            "why": "Negative control. Correct behaviour is refusal. In 2.1 retrieval "
                   "returned five irrelevant chunks and grounding held only because of "
                   "the system prompt.",
            "question": "What is the current market capitalization of Nvidia?",
            "grading_notes": (
                "The answer must state that the provided documents do not contain this "
                "information. It must NOT supply a figure or estimate from outside the "
                "documents."
            ),
        },
    ]


# ─── Baseline evaluation ─────────────────────────────────────────────
def run_evaluation() -> list[dict]:
    llm = make_llm(ANSWER_MODEL)
    judge_llm = make_llm(JUDGE_MODEL)
    eval_set = my_eval_set()
    results = []

    header = (f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}  |  "
              f"top_k={TOP_K}  |  answerer={ANSWER_MODEL}  judge={JUDGE_MODEL}")
    print(header + "\n")
    log("RUN START", header)

    for item in eval_set:
        hits = retrieve(item["question"], TOP_K)
        ans = (answer(llm, item["question"], [cid for cid, _ in hits])
               if hits else "(no documents retrieved)")
        verdict = judge(judge_llm, ans, item["grading_notes"])
        matched = verdict == item["expectation"]

        sources = sorted({cid.split("#")[0] for cid, _ in hits})

        print("=" * 78)
        print(f"{item['id']}  [expected {item['expectation'].upper()}]")
        print(f"  Q: {item['question']}")
        print(f"  retrieved {len(hits)} chunks from {len(sources)} distinct paper(s):")
        for cid, score in hits:
            print(f"      {score:6.3f}  {cid}")
        print(f"  VERDICT: {verdict.upper()}"
              f"   {'(as predicted)' if matched else '(UNEXPECTED)'}")
        print(f"  answer: {ans}\n")

        log(f"{item['id']}: {item['question']}",
            f"expected={item['expectation']}\nverdict={verdict}\n"
            f"distinct_sources={len(sources)}\nretrieved={hits}\nanswer={ans}")

        results.append({**item, "verdict": verdict, "hits": hits,
                        "distinct_sources": len(sources), "answer": ans,
                        "matched_prediction": matched})

    passes = sum(r["verdict"] == "pass" for r in results)
    predicted = sum(r["matched_prediction"] for r in results)

    print("=" * 78)
    print(f"  Baseline pass rate:        {passes}/{len(results)}")
    print(f"  Matched prior prediction:  {predicted}/{len(results)}")
    print("=" * 78)

    print("\n  Per-item summary")
    print(f"  {'id':<26} {'expected':<10} {'actual':<8} {'papers':<8} match")
    for r in results:
        print(f"  {r['id']:<26} {r['expectation']:<10} {r['verdict']:<8} "
              f"{r['distinct_sources']:<8} {'yes' if r['matched_prediction'] else 'NO'}")

    log("SUMMARY", f"pass_rate={passes}/{len(results)} "
                   f"predicted={predicted}/{len(results)}")
    return results


# ─── Framework validation ────────────────────────────────────────────
def validate_framework() -> None:
    """Confirm the judge fails a deliberately false answer, not just passes good ones."""
    llm = make_llm(ANSWER_MODEL)
    judge_llm = make_llm(JUDGE_MODEL)

    q = ("What is storyboard programming and how does it synthesize data structure "
         "manipulations?")
    notes = my_eval_set()[0]["grading_notes"]

    hits = retrieve(q, TOP_K)
    good = answer(llm, q, [cid for cid, _ in hits])

    manipulated = (
        "Storyboard programming is a video editing technique in which animators arrange "
        "frames on a timeline. It uses a convolutional neural network trained on Pixar "
        "films to interpolate between keyframes, and has no connection to program "
        "synthesis or data structures."
    )

    subtle = (
        "Storyboard programming is an approach to program synthesis developed at MIT. "
        "It helps programmers write correct code more easily and has been evaluated on "
        "a range of benchmarks with encouraging results."
    )

    print("\n" + "=" * 78)
    print("  FRAMEWORK VALIDATION")
    print("=" * 78)

    for label, text, expected in (
        ("correct answer", good, "PASS"),
        ("fabricated answer", manipulated, "FAIL"),
        ("vague but not false", subtle, "FAIL"),
    ):
        v = judge(judge_llm, text, notes)
        flag = "ok" if v.upper() == expected else "*** MISMATCH ***"
        print(f"  {label:<22} -> {v.upper():<5} (expected {expected})  {flag}")
        log("FRAMEWORK VALIDATION", f"{label}: verdict={v} expected={expected}")

    print("\n  The framework is sound if it passes the correct answer and fails both")
    print("  wrong ones. The third case is the harder test: an answer that is not")
    print("  false, merely empty of the required content.")


# ─── Parser robustness ───────────────────────────────────────────────
def parser_robustness_check() -> None:
    """Demonstrate the substring-matching flaw in the starter's verdict parser.

    The starter used:  return "pass" if "pass" in verdict else "fail"
    Any judge reply containing the substring 'pass' scores as a pass, including
    'the answer does not pass'. This is a silent failure that inflates pass rates.
    """
    print("\n" + "=" * 78)
    print("  VERDICT PARSER ROBUSTNESS")
    print("=" * 78)

    simulated_replies = [
        "pass",
        "fail",
        "Fail.",
        "The answer does not pass the grading notes.",
        "This would not pass a strict reading.",
        "passable, but incomplete",
        "",
    ]

    def starter_parser(v: str) -> str:
        v = v.strip().lower()
        return "pass" if "pass" in v else "fail"

    def strict_parser(v: str) -> str:
        v = v.strip().lower()
        first = v.split()[0].strip(".,:;'\"") if v.split() else ""
        return first if first in ("pass", "fail") else "fail"

    print(f"  {'judge reply':<46} {'starter':<10} {'strict':<10} differs")
    for reply in simulated_replies:
        a, b = starter_parser(reply), strict_parser(reply)
        shown = f'"{reply}"' if reply else "(empty)"
        print(f"  {shown:<46} {a:<10} {b:<10} {'YES' if a != b else ''}")

    print("\n  Rows marked YES are cases where the starter's parser would have")
    print("  recorded a PASS for a judge reply that clearly means fail.")
    log("PARSER CHECK", "substring parser vs first-token parser compared")


# ─── Cross-model sanity check ────────────────────────────────────────
def cross_model_check(alt_judge: str = "anthropic/claude-3.5-haiku") -> None:
    """Re-judge the eval set with a different judge model.

    Using the same model to write and grade an answer risks self-preference bias.
    Agreement between two independent judges is evidence the metric is measuring
    the answer rather than the model's opinion of itself.
    """
    llm = make_llm(ANSWER_MODEL)
    judge_a = make_llm(JUDGE_MODEL)
    judge_b = make_llm(alt_judge)

    print("\n" + "=" * 78)
    print(f"  CROSS-MODEL JUDGE AGREEMENT  ({JUDGE_MODEL} vs {alt_judge})")
    print("=" * 78)

    agree = 0
    items = my_eval_set()
    for item in items:
        hits = retrieve(item["question"], TOP_K)
        ans = (answer(llm, item["question"], [cid for cid, _ in hits])
               if hits else "(no documents retrieved)")
        va = judge(judge_a, ans, item["grading_notes"])
        vb = judge(judge_b, ans, item["grading_notes"])
        agree += va == vb
        print(f"  {item['id']:<26} {va:<6} {vb:<6} {'agree' if va == vb else 'DISAGREE'}")
        log("CROSS-MODEL", f"{item['id']}: {JUDGE_MODEL}={va} {alt_judge}={vb}")

    print(f"\n  Agreement: {agree}/{len(items)}")
    if agree < len(items):
        print("  Disagreements indicate the grading notes are ambiguous, or that one")
        print("  judge is applying a different strictness threshold. Both are findings.")


# ─── Entry point ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint 3.1 evaluation.")
    ap.add_argument("--skip-validation", action="store_true",
                    help="Run only the baseline evaluation.")
    ap.add_argument("--cross-model", action="store_true",
                    help="Also re-judge with a second judge model (extra API calls).")
    ap.add_argument("--alt-judge", default="anthropic/claude-3.5-haiku",
                    help="Model to use for the cross-model check.")
    args = ap.parse_args()

    run_evaluation()

    if not args.skip_validation:
        validate_framework()
        parser_robustness_check()

    if args.cross_model:
        cross_model_check(args.alt_judge)

    print(f"\nLogged to {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()