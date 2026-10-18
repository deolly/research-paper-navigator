"""inspect_probes.py — read checkpoint_6_1_results.json and show which probes failed.

Written as a file rather than a `python -c` one-liner because PowerShell mangles
nested quotes and f-strings on the command line.

Usage
    python inspect_probes.py              # rate table, every non-zero cell
    python inspect_probes.py --text       # plus the model's actual reply
    python inspect_probes.py --all        # every cell, including the clean ones
    python inspect_probes.py --chars 1200 # longer excerpts with --text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("checkpoint_6_1_results.json")


def sections(d: dict) -> list[tuple[str, dict]]:
    """Both --probe and --ladder results, whichever the file happens to hold."""
    out = []
    if "security" in d:
        out.append(("--probe", d["security"]))
    for m in d.get("ladder", []):
        out.append((f"ladder:{m.get('tier', '?')}", m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="print the reply text too")
    ap.add_argument("--all", action="store_true", help="include cells that held")
    ap.add_argument("--chars", type=int, default=700)
    args = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found — run a --probe or --ladder pass first.")

    d = json.loads(RESULTS.read_text(encoding="utf-8"))
    blocks = sections(d)
    if not blocks:
        raise SystemExit("No probe or ladder results in the file.")

    print("=" * 86)
    print(f"  {'model':<22} {'probe':<26} {'baseline':<10} {'hardened':<10} trials (b | h)")
    print("=" * 86)

    for _tag, block in blocks:
        model = block.get("model", "?")
        for pid, v in block.get("detail", {}).items():
            # Older result files stored a single boolean per cell instead of a rate.
            br = v.get("baseline_rate")
            hr = v.get("hardened_rate")
            if br is None:
                br = float(bool(v["baseline"].get("compromised")))
                hr = float(bool(v["hardened"].get("compromised")))
            if not (br or hr) and not args.all:
                continue
            bt = v.get("baseline_trials", [])
            ht = v.get("hardened_trials", [])
            fmt = lambda t: "".join("X" if x else "." for x in t) or "-"
            print(f"  {model:<22} {pid:<26} {br:<10.1f} {hr:<10.1f} "
                  f"{fmt(bt)} | {fmt(ht)}")

    if not args.text:
        print("\n  X = compromised on that run, . = held.  Add --text for the replies.")
        return

    for _tag, block in blocks:
        model = block.get("model", "?")
        for pid, v in block.get("detail", {}).items():
            for variant in ("baseline", "hardened"):
                rate = v.get(f"{variant}_rate")
                if rate is None:
                    rate = float(bool(v[variant].get("compromised")))
                if not rate and not args.all:
                    continue
                print("\n" + "=" * 86)
                print(f"  {model} | {pid} | {variant}  rate={rate:.1f}"
                      f"  trials={v.get(f'{variant}_trials', '?')}")
                print("-" * 86)
                print(v[variant]["output"][:args.chars])


if __name__ == "__main__":
    main()
