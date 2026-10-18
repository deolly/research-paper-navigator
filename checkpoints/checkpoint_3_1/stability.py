from capstone_checkpoint_3_1_evaluation import *
from collections import Counter

llm = make_llm()
jl = make_llm(JUDGE_MODEL)
item = my_eval_set()[3]

hits = retrieve(item["question"], 5)

print(f"Testing: {item['id']}\n")

# A: full pipeline — answer regenerated each time
print("A. Answer regenerated each run (answerer + judge both vary)")
va = []
for i in range(5):
    ans = answer(llm, item["question"], [c for c, _ in hits])
    v = judge(jl, ans, item["grading_notes"])
    va.append(v)
    print(f"   run {i+1}: {v}")
print(f"   -> {Counter(va)}\n")

# B: one fixed answer, judged repeatedly — isolates judge variance
print("B. Same answer judged 5 times (judge variance only)")
fixed = answer(llm, item["question"], [c for c, _ in hits])
vb = [judge(jl, fixed, item["grading_notes"]) for _ in range(5)]
for i, v in enumerate(vb, 1):
    print(f"   run {i}: {v}")
print(f"   -> {Counter(vb)}")