from capstone_checkpoint_3_1_evaluation import *
from collections import Counter

llm = make_llm(); jl = make_llm(JUDGE_MODEL)
for item in my_eval_set():
    hits = retrieve(item["question"], 5)
    vs = []
    for _ in range(3):
        ans = answer(llm, item["question"], [c for c, _ in hits])
        vs.append(judge(jl, ans, item["grading_notes"]))
    c = Counter(vs)
    stable = "stable" if len(c) == 1 else "UNSTABLE"
    print(f"{item['id']:<26} {dict(c)}  {stable}")