from langchain_core.messages import HumanMessage, SystemMessage

from capstone_checkpoint_3_1_evaluation import *

llm = make_llm()
jl = make_llm(JUDGE_MODEL)
item = my_eval_set()[3]

hits = retrieve(item["question"], 5)
ans = answer(llm, item["question"], [c for c, _ in hits])

print("=" * 78)
print("GRADING NOTES")
print("=" * 78)
print(item["grading_notes"])

print("\n" + "=" * 78)
print("ANSWER")
print("=" * 78)
print(ans)

base = "ANSWER:\n" + ans + "\n\nGRADING NOTES:\n" + item["grading_notes"]

verdict_first = base + (
    "\n\nReply pass or fail, then state in one sentence exactly which "
    "requirement was or was not met."
)

reasoning_first = base + (
    "\n\nFirst, state in one sentence whether each requirement is met. "
    "Then, on a new final line, write only PASS or FAIL."
)

for label, prompt in (("VERDICT FIRST", verdict_first),
                      ("REASONING FIRST", reasoning_first)):
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print(jl.invoke([
        SystemMessage(content="You are a strict evaluator."),
        HumanMessage(content=prompt),
    ]).content)