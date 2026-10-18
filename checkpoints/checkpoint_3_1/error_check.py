from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from capstone_checkpoint_3_1_evaluation import *


def judge_verbose(llm, answer_text, grading_notes):
    try:
        raw = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
        ]).content.strip().lower()
    except Exception as e:
        print(f"      !! JUDGE API ERROR: {type(e).__name__}: {e}")
        return "error"
    first = raw.split()[0].strip(".,:;'\"") if raw.split() else ""
    if first not in ("pass", "fail"):
        print(f"      !! NON-VERDICT REPLY: {raw[:120]!r}")
        return "error"
    return first


N = 10
llm = make_llm()
jl = make_llm(JUDGE_MODEL)

# E4 is the item under investigation; E1 is the control that measured 10/10
for idx in (3, 0):
    item = my_eval_set()[idx]
    hits = retrieve(item["question"], 5)
    print(f"\n{item['id']}  ({N} runs)")

    verdicts = []
    for i in range(N):
        try:
            ans = answer(llm, item["question"], [c for c, _ in hits])
        except Exception as e:
            print(f"      !! ANSWER API ERROR: {type(e).__name__}: {e}")
            verdicts.append("error")
            continue
        verdicts.append(judge_verbose(jl, ans, item["grading_notes"]))

    c = Counter(verdicts)
    errors = c.get("error", 0)
    scored = N - errors
    rate = c.get("pass", 0) / scored if scored else 0
    print(f"   {dict(c)}")
    print(f"   pass rate over completed runs: {c.get('pass', 0)}/{scored} = {rate:.0%}")
    if errors:
        print(f"   *** {errors} run(s) failed to complete — earlier numbers understate the pass rate")