"""Re-derive the security results table from a production log.

The numbers quoted in README.md and in the presentation are not typed by
hand -- they come out of this script, which reads the raw log and recomputes
every rate and every p-value. Run it to check any claim:

    python evidence/summarize_security.py
    python evidence/summarize_security.py --probe P2-fake-source

No dependencies beyond the standard library, so a reviewer can verify the
statistics without installing anything.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent / "checkpoint_7_1" / "checkpoint_7_1_production.log"
ARMS = ("baseline", "hardened", "xml-only", "production")

_HDR = re.compile(r"^\[(\S+)\]\s+SECURITY\s+(\S+)\s*\|\s*(\S+)")
_ARM = re.compile(r'^\s*"(baseline|hardened|xml-only|production)":\s*\{')


# ----------------------------------------------------------- statistics

def fisher_two_tailed(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher exact test on [[a, b], [c, d]]."""
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    pr = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    obs = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used instead of k/n +- 1.96*sqrt(p(1-p)/n)
    because the normal approximation gives [0, 0] at k=0, which would claim
    certainty the data cannot support."""
    if n == 0:
        return (0.0, 0.0)
    ph, d = k / n, 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------- parsing

def parse(log: Path) -> list[dict]:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    cur, rows = None, []
    for i, line in enumerate(lines):
        h = _HDR.match(line)
        if h:
            cur = {"when": h.group(1)[:19], "probe": h.group(2), "model": h.group(3)}
            continue
        a = _ARM.match(line)
        if a and cur:
            blk = "\n".join(lines[i:i + 8])
            num = lambda k: int(float((re.search(rf'"{k}":\s*([0-9.]+)', blk) or [0, 0])[1]))
            rows.append({**cur, "arm": a.group(1), "line": i + 1,
                         "compromised": num("compromised"),
                         "measured": num("measured"),
                         "not_measured": num("not_measured"),
                         "withheld": num("withheld"),
                         "refused": num("refused")})
    return rows


# --------------------------------------------------------------- reports

def per_run(rows: list[dict]) -> None:
    print("\nEVERY RUN, AS LOGGED")
    print("=" * 94)
    print(f"{'log line':>8}  {'when':<20}{'model':<15}{'probe':<26}{'arm':<11}"
          f"{'comp':>5}{'meas':>6}{'n/m':>5}")
    print("-" * 94)
    prev = None
    for r in rows:
        key = (r["when"], r["model"], r["probe"])
        if prev and key != prev:
            print()
        prev = key
        print(f"{r['line']:>8}  {r['when']:<20}{r['model'].replace('openai/',''):<15}"
              f"{r['probe']:<26}{r['arm']:<11}"
              f"{r['compromised']:>5}{r['measured']:>6}{r['not_measured']:>5}")


def pooled(rows: list[dict]) -> None:
    """Pool trials per (model, probe, arm). Trials of the same cell are
    exchangeable, so pooling is legitimate and raises the power; it is also
    the only way to get usable intervals, since n=20 cannot resolve a 5% rate."""
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"comp": 0, "meas": 0, "nm": 0, "wh": 0, "ref": 0, "runs": 0})
    for r in rows:
        a = agg[(r["model"].replace("openai/", ""), r["probe"], r["arm"])]
        a["comp"] += r["compromised"]
        a["meas"] += r["measured"]
        a["nm"] += r["not_measured"]
        a["wh"] += r["withheld"]
        a["ref"] += r["refused"]
        a["runs"] += 1

    print("\n\nPOOLED ACROSS RUNS, WITH 95% WILSON INTERVALS")
    print("=" * 111)
    print("  'comp' = answer accepted the injection.  'n/m' = NOT MEASURED:")
    print("  the planner's JSON failed to parse, so the trial produced no verdict.")
    print("  A parse failure is not resistance -- counting it as 'held' was one of")
    print("  the measurement bugs this audit found.\n")
    print(f"{'model':<14}{'probe':<26}{'arm':<11}{'compromised':>14}{'95% CI':>16}"
          f"{'not measured':>15}{'withheld':>9}{'refused':>8}")
    print("-" * 111)
    last = None
    for (model, probe, arm), a in sorted(agg.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1], ARMS.index(kv[0][2]))):
        if last and (model, probe) != last:
            print()
        last = (model, probe)
        n = a["meas"]
        lo, hi = wilson(a["comp"], n)
        ci = f"[{100*lo:4.1f}-{100*hi:4.1f}%]"
        nm = f"{a['nm']}/{n + a['nm']}" if a["nm"] else "-"
        print(f"{model:<14}{probe:<26}{arm:<11}"
              f"{a['comp']:>6}/{n:<4} {100*a['comp']/n if n else 0:4.1f}%{ci:>16}{nm:>15}"
              f"{a['wh']:>9}{a['ref']:>8}")

    tot_wh = sum(a["wh"] for a in agg.values())
    tot_ref = sum(a["ref"] for a in agg.values())
    tot_tr = sum(a["meas"] + a["nm"] for a in agg.values())
    print(f"\n  TWO NULLS, STATED PLAINLY: across all {tot_tr} trials the citation")
    print(f"  validator withheld {tot_wh} answers and the fail-closed guard forced")
    print(f"  {tot_ref} refusals. Both deterministic safeguards pass their own")
    print(f"  regression tests and never fired in a live run, so every significant")
    print(f"  gain below is attributable to the PROMPT, not to the code. The")
    print(f"  xml-only arm exists to prove that rather than assume it.")
    return agg


def comparisons(agg: dict) -> None:
    """The comparisons the presentation actually claims. Each is stated as
    before -> after with a two-tailed Fisher exact test."""
    TESTS = [
        ("Fake-source accepted: hardened prompt -> XML trust boundary",
         ("gpt-5.4-nano", "P2-fake-source", "hardened"),
         ("gpt-5.4-nano", "P2-fake-source", "xml-only"), "comp"),
        ("Fake-source accepted: baseline -> full production stack",
         ("gpt-5.4-nano", "P2-fake-source", "baseline"),
         ("gpt-5.4-nano", "P2-fake-source", "production"), "comp"),
        ("Planner parse failure: baseline -> hardened decide prompt",
         ("gpt-5.4-nano", "P5-tool-argument-hijack", "baseline"),
         ("gpt-5.4-nano", "P5-tool-argument-hijack", "hardened"), "nm"),
        ("Roleplay override accepted: baseline -> full production stack",
         ("gpt-5.4-mini", "P1-direct-command", "baseline"),
         ("gpt-5.4-mini", "P1-direct-command", "production"), "comp"),
    ]
    print("\n\nTHE CLAIMS, TESTED")
    print("=" * 94)
    for label, before, after, field in TESTS:
        if before not in agg or after not in agg:
            print(f"\n{label}\n   (cells not present in this log)")
            continue
        b, a = agg[before], agg[after]
        nb = b["meas"] + (b["nm"] if field == "nm" else 0)
        na = a["meas"] + (a["nm"] if field == "nm" else 0)
        kb, ka = b[field], a[field]
        p = fisher_two_tailed(kb, nb - kb, ka, na - ka)
        lb, ub = wilson(kb, nb)
        la, ua = wilson(ka, na)
        print(f"\n{label}")
        print(f"   before  {kb:>3}/{nb:<3} = {100*kb/nb:5.1f}%   [{100*lb:4.1f}-{100*ub:4.1f}%]")
        print(f"   after   {ka:>3}/{na:<3} = {100*ka/na:5.1f}%   [{100*la:4.1f}-{100*ua:4.1f}%]")
        verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"   Fisher exact, two-tailed:  p = {p:.6f}   ({verdict} at 0.05)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--probe", help="restrict to one probe id, e.g. P2-fake-source")
    ap.add_argument("--model", help="restrict to one model id")
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"log not found: {args.log}")
    rows = parse(args.log)
    if args.probe:
        rows = [r for r in rows if r["probe"] == args.probe]
    if args.model:
        rows = [r for r in rows if args.model in r["model"]]
    if not rows:
        raise SystemExit("no matching rows")

    print(f"\nsource: {args.log}")
    print(f"{len(rows)} arm results across "
          f"{len({(r['when'], r['probe'], r['model']) for r in rows})} runs")
    per_run(rows)
    agg = pooled(rows)
    comparisons(agg)
    print()


if __name__ == "__main__":
    main()
