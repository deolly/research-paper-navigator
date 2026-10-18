from capstone_checkpoint_3_1_evaluation import *
from collections import Counter

N = 10
llm = make_llm(); jl = make_llm(JUDGE_MODEL)

print(f"{'item':<26} {'pass rate':<12} verdicts")
totals = []
for item in my_eval_set():
    hits = retrieve(item["question"], 5)
    vs = []
    for _ in range(N):
        ans = answer(llm, item["question"], [c for c, _ in hits])
        vs.append(judge(jl, ans, item["grading_notes"]))
    p = vs.count("pass") / N
    totals.append(p)
    print(f"{item['id']:<26} {p:>5.0%}        {dict(Counter(vs))}")

print(f"\nExpected pass rate: {sum(totals):.1f}/5")