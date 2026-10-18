from capstone_checkpoint_3_1_evaluation import *

llm = make_llm()
jl = make_llm(JUDGE_MODEL)
item = my_eval_set()[3]

hits = retrieve(item["question"], 5)
ans = answer(llm, item["question"], [c for c, _ in hits])

print("\n--- GRADING NOTES ---")
print(item["grading_notes"])

print("\n--- ANSWER ---")
print(ans)

print("\n--- RAW JUDGE REPLY ---")
print(judge_raw(jl, ans, item["grading_notes"]))

print("\n--- PARSED VERDICT ---")
print(judge(jl, ans, item["grading_notes"]))

r"""Printing the notes and answer alongside the raw reply matters here — the answer is regenerated on this run, 
and at temperature 0.2 it may differ slightly from the one you saw. You need to judge the reply against the 
answer that was actually graded, not the earlier one.

If the judge just says "fail" with no reasoning, add this to get an explanation but remove the # from line 28 through 37.
That tells you directly whether it objected to "systems or papers" versus the trailing offer.
"""
#from langchain_core.messages import HumanMessage, SystemMessage

#explain = jl.invoke([
#    SystemMessage(content="You are a strict evaluator."),
#    HumanMessage(content=f"ANSWER:\n{ans}\n\nGRADING NOTES:\n{item['grading_notes']}\n\n"
#                         f"Does the answer satisfy the notes? Reply pass or fail, then "
#                         f"explain in one sentence which requirement was or was not met."),
#]).content
#print("\n--- EXPLAINED ---")
#print(explain)