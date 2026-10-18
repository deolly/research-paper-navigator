"""capstone_eval_4_1.py — comparative evaluation: baseline vs advanced retrieval.

Re-runs the Checkpoint 3.1 evaluation set against both systems and reports
per-item pass RATES over N_RUNS, not single verdicts. Checkpoint 3.1 established
that a single run is not a measurement: one run gave 3/5 while ten runs per item
gave an expected 3.1/5, and two of the original readings were sampling artefacts.

Fairness: both systems send exactly TOP_K chunks to the answering model
(baseline 5 primary; advanced 3 primary + 2 graph context), so any difference
in outcome is not a difference in context size.

Usage
    python capstone_eval_4_1.py                 # full run, N_RUNS per item
    python capstone_eval_4_1.py --runs 3        # cheaper
    python capstone_eval_4_1.py --items E3 E4   # just the two failing items
    python capstone_eval_4_1.py --mode decompose  # evaluate a different variant
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import capstone_advanced as adv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-5.4-mini"
N_RUNS = 5
LOG_PATH = Path("checkpoint_4_1_evaluation.log")
RESULTS_JSON = Path("checkpoint_4_1_results.json")

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES "
    "describing what a correct answer must contain. First state in one sentence "
    "whether each requirement is met. Then, on a new final line, write only "
    "PASS or FAIL."
)


def make_llm(model: str = JUDGE_MODEL, temperature: float = 0.0) -> ChatOpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — check your .env file.")
    return ChatOpenAI(model=model, temperature=temperature,
                      api_key=key, base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 74}\n")


# ─── The metric ──────────────────────────────────────────────────────
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    """Reasoning-first judging, verdict taken from the LAST token.

    Checkpoint 3.1 observed a reply reading 'pass — ...it does not name at least
    two systems as required': a correct rationale with a contradictory leading
    token, which first-token extraction scored as a pass. Asking for reasoning
    before the verdict, and reading the final line, avoids that ordering.
    Anything unparseable is recorded as 'error' rather than silently failing,
    so a failed call is never mistaken for a failed system.
    """
    try:
        raw = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=f"ANSWER:\n{answer_text}\n\n"
                                 f"GRADING NOTES:\n{grading_notes}"),
        ]).content.strip()
    except Exception as e:
        print(f"        !! JUDGE ERROR: {type(e).__name__}")
        return "error"

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "error"
    last = lines[-1].strip("*_ .:;'\"").lower()
    if last.endswith("pass"):
        return "pass"
    if last.endswith("fail"):
        return "fail"
    print(f"        !! NON-VERDICT FINAL LINE: {lines[-1][:80]!r}")
    return "error"


# ─── The evaluation set, carried over unchanged from Checkpoint 3.1 ──
def eval_set() -> list[dict]:
    return [
        {
            "id": "E1",
            "name": "rare coined term",
            "baseline_3_1": 1.00,
            "question": "What is storyboard programming and how does it synthesize "
                        "data structure manipulations?",
            "grading_notes": (
                "The answer must state that storyboard programming synthesizes "
                "low-level imperative data-structure code from a storyboard, and "
                "must mention at least two of: abstract input-output scenarios, "
                "fold/unfold definitions, a skeleton of the loop structure, or "
                "encoding as a constraint satisfaction problem."
            ),
        },
        {
            "id": "E2",
            "name": "accessible paraphrase",
            "baseline_3_1": 1.00,
            "question": "How can a system automatically help students fix mistakes "
                        "in their programming assignments?",
            "grading_notes": (
                "The answer must describe using an instructor's reference "
                "implementation together with an error model of likely student "
                "mistakes to compute minimal corrections to the student's program."
            ),
        },
        {
            "id": "E3",
            "name": "vocabulary gap",
            "baseline_3_1": 0.00,
            "question": "How can a programmer leave gaps in their code for the "
                        "computer to fill in automatically?",
            "grading_notes": (
                "The answer must explain that the programmer writes a partial "
                "program — a sketch or template — containing holes, and that a "
                "synthesizer automatically fills those holes so the program "
                "satisfies a specification."
            ),
        },
        {
            "id": "E4",
            "name": "multi-document aggregation",
            "baseline_3_1": 0.10,
            "question": "Which different approaches to learning reusable libraries "
                        "of program abstractions appear in this corpus, and how do "
                        "they differ from each other?",
            "grading_notes": (
                "The answer must name at least two systems by name (for example "
                "Stitch, DreamCoder, or a named author's approach) and state a "
                "specific difference between them. Naming abstract categories of "
                "approach without identifying specific systems does not satisfy this."
            ),
        },
        {
            "id": "E5",
            "name": "out-of-corpus control",
            "baseline_3_1": 1.00,
            "question": "What is the current market capitalization of Nvidia?",
            "grading_notes": (
                "The answer must state that the provided documents do not contain "
                "this information. It must NOT supply a figure or estimate from "
                "outside the documents."
            ),
        },
    ]


# ─── One measured cell: system x item, repeated N times ──────────────
def measure(mode: str, item: dict, runs: int, ans_llm, judge_llm) -> dict:
    verdicts, papers_seen, sub_query_log = [], [], []

    for i in range(runs):
        primary, context, subqs = adv.retrieve(mode, item["question"], ans_llm,
                                               verbose=False)
        if mode in ("decompose", "both") and subqs != [item["question"]]:
            sub_query_log.append(subqs)

        srcs = {c.split("#")[0] for c, _, _ in primary}
        srcs |= {c.split("#")[0] for c, _, _, _ in context}
        papers_seen.append(len(srcs))

        try:
            answer = adv.answer(ans_llm, item["question"], primary, context)
        except Exception as e:
            print(f"        !! ANSWER ERROR: {type(e).__name__}")
            verdicts.append("error")
            continue

        v = judge(judge_llm, answer, item["grading_notes"])
        verdicts.append(v)
        print(f"        run {i+1}: {v}")

        log(f"{mode} | {item['id']} | run {i+1}",
            f"primary={[c for c,_,_ in primary]}\n"
            f"context={[(c,p) for c,_,_,p in context]}\n"
            f"distinct_papers={len(srcs)}\nverdict={v}\nanswer={answer}")

    counts = Counter(verdicts)
    completed = runs - counts.get("error", 0)
    rate = counts.get("pass", 0) / completed if completed else 0.0

    return {
        "verdicts": verdicts,
        "pass_rate": rate,
        "completed": completed,
        "errors": counts.get("error", 0),
        "avg_distinct_papers": sum(papers_seen) / len(papers_seen) if papers_seen else 0,
        "sub_queries": sub_query_log[:2],
    }


# ─── Main ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--mode", default="both",
                    choices=["decompose", "graph", "both"],
                    help="which advanced variant to compare against baseline")
    ap.add_argument("--items", nargs="*", help="subset of item ids, e.g. E3 E4")
    args = ap.parse_args()

    items = eval_set()
    if args.items:
        wanted = {x.upper() for x in args.items}
        items = [i for i in items if i["id"] in wanted]

    ans_llm = adv.make_llm()
    judge_llm = make_llm()

    print("=" * 78)
    print(f"  Checkpoint 4.1 — baseline vs '{args.mode}'   n={args.runs} per item")
    print(f"  Both systems send {adv.TOP_K} chunks to the answering model.")
    print("=" * 78)

    results = {}
    for item in items:
        print(f"\n{item['id']} — {item['name']}")
        print(f"  {item['question'][:74]}")
        results[item["id"]] = {"item": item}
        for mode in ("baseline", args.mode):
            print(f"    {mode}:")
            results[item["id"]][mode] = measure(mode, item, args.runs,
                                                ans_llm, judge_llm)

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"  {'item':<6} {'3.1':<7} {'baseline':<10} {args.mode:<10} "
          f"{'delta':<8} {'papers b/a'}")

    base_total = adv_total = 0.0
    for iid, r in results.items():
        b, a = r["baseline"], r[args.mode]
        base_total += b["pass_rate"]
        adv_total += a["pass_rate"]
        delta = a["pass_rate"] - b["pass_rate"]
        arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
        print(f"  {iid:<6} {r['item']['baseline_3_1']:<7.0%} "
              f"{b['pass_rate']:<10.0%} {a['pass_rate']:<10.0%} "
              f"{arrow}{abs(delta):<7.0%} "
              f"{b['avg_distinct_papers']:.1f} / {a['avg_distinct_papers']:.1f}")

    n = len(results)
    print(f"\n  Expected pass rate  baseline: {base_total:.1f}/{n}"
          f"   {args.mode}: {adv_total:.1f}/{n}")

    errs = sum(r[m]["errors"] for r in results.values() for m in ("baseline", args.mode))
    if errs:
        print(f"  *** {errs} run(s) did not complete — rates are over completed runs")

    # Sub-queries actually generated, as evidence for the writeup
    print("\n  Sub-queries generated (first example per item):")
    for iid, r in results.items():
        sq = r[args.mode].get("sub_queries")
        if sq:
            print(f"    {iid}:")
            for s in sq[0]:
                print(f"        - {s}")

    RESULTS_JSON.write_text(json.dumps(
        {k: {m: v for m, v in r.items() if m != "item"} for k, r in results.items()},
        indent=2), encoding="utf-8")
    print(f"\n  Results written to {RESULTS_JSON}")
    print(f"  Full transcripts in {LOG_PATH}")


if __name__ == "__main__":
    main()
