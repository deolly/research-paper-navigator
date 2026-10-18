"""Research Paper Navigator -- single entry point.

The system grew one checkpoint at a time, so its capabilities live in eight
modules under src/, each with its own argparse CLI. That is fine to develop
against and hostile to review: nothing tells a newcomer that
`capstone_production.py --probe --repeat 30` is the interesting command.

This file is a thin dispatcher. It adds no behaviour of its own -- it sets
sys.argv and calls the module's own main(), so every command here is exactly
reproducible by hand, and the underlying scripts stay independently runnable.

    python navigator.py                     list the commands
    python navigator.py selftest            offline regression tests, no API calls
    python navigator.py ask "..."           answer one question
    python navigator.py security            injection probes, four arms
    python navigator.py cost                token accounting
    python navigator.py evaluate            retrieval + answer quality
    python navigator.py graph               build/inspect the paper graph
    python navigator.py demo                print the presentation run order
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _run(module_name: str, argv: list[str]) -> None:
    """Call a module's own main() with the argv it expects."""
    import importlib

    mod = importlib.import_module(module_name)
    printable = f"python src/{module_name}.py " + " ".join(
        f'"{a}"' if " " in a else a for a in argv
    )
    print(f"\n>>> {printable}\n")
    saved = sys.argv
    sys.argv = [f"{module_name}.py", *argv]
    try:
        mod.main()
    finally:
        sys.argv = saved


def _need_corpus() -> None:
    from paths import corpus_dir

    corpus_dir(required=True)


# --------------------------------------------------------------- commands

def cmd_selftest(rest: list[str]) -> None:
    """Offline regression tests: sanitizer, citation validator, router, cache.

    No API calls, no corpus, no cost. These are the labelled test sets that
    caught four measurement bugs in the instruments themselves.
    """
    _run("capstone_production", ["--offline", *rest])


def cmd_ask(rest: list[str]) -> None:
    """Answer one question through the full agentic loop."""
    if not rest:
        sys.exit('usage: python navigator.py ask "your question"')
    _need_corpus()
    _run("capstone_agent", ["--query", rest[0], *rest[1:]])


def cmd_security(rest: list[str]) -> None:
    """Prompt-injection probes across four arms.

    baseline / hardened / xml-only / production. The xml-only arm exists to
    separate what the prompt fixed from what the code fixed; without it the
    attribution is guesswork. Default 30 trials per cell -- n=5 cannot
    detect a 30% failure rate.
    """
    _need_corpus()
    if not any(a.startswith("--repeat") for a in rest):
        rest = [*rest, "--repeat", "30"]
    _run("capstone_production", ["--probe", *rest])


def cmd_cost(rest: list[str]) -> None:
    """Per-role token accounting and the direct-vs-agent comparison."""
    _need_corpus()
    _run("capstone_production", ["--cost", *rest])


def cmd_audit(rest: list[str]) -> None:
    """The checkpoint 6.1 security audit: five injection surfaces, model ladder."""
    _need_corpus()
    _run("capstone_security_cost", ["--probe", *rest])


def cmd_evaluate(rest: list[str]) -> None:
    """Retrieval and answer-quality evaluation.

    `--checkpoint 3` is the LLM-as-judge evaluation with a pinned judge;
    `--checkpoint 4` is the advanced-retrieval comparison. Default: both.
    """
    which = "both"
    if "--checkpoint" in rest:
        i = rest.index("--checkpoint")
        which = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    _need_corpus()
    if which in ("3", "both"):
        _run("capstone_eval_3_1", rest)
    if which in ("4", "both"):
        _run("capstone_eval_4_1", rest)


def cmd_graph(rest: list[str]) -> None:
    """Build or inspect the paper graph (137 nodes, 482 edges)."""
    _need_corpus()
    _run("capstone_graph", rest)


def cmd_demo(rest: list[str]) -> None:
    """Print the run order used for the recorded presentation."""
    print(textwrap.dedent("""
        Presentation run order
        ======================
        Each line is a separate run. Times assume the prebuilt index in
        chroma_papers/; without it add ~10 minutes to the first corpus run.

          1.  python navigator.py selftest
              Offline. Sanitizer 0/18 false positives, validator 7/7,
              router 11/12, cache separability. Free, ~2 seconds.

          2.  python navigator.py security --model openai/gpt-5.4-nano
              Four arms x 30 trials. The ATTRIBUTION block at the end is the
              headline result -- do not crop it. ~15 minutes, costs money.

          3.  python navigator.py cost
              Direct 1,381 vs agent 3,312 tokens. ~3 minutes.

          4.  python navigator.py evaluate --checkpoint 3
              Judged answer quality with a pinned judge model.

          5.  python navigator.py ask "How do these papers define agentic RAG?"
              One live end-to-end answer with citations.

        Frozen output from every one of these is in evidence/, so the
        presentation can be recorded without spending the API budget again.
    """).strip() + "\n")


COMMANDS = {
    "selftest": cmd_selftest,
    "ask": cmd_ask,
    "security": cmd_security,
    "cost": cmd_cost,
    "audit": cmd_audit,
    "evaluate": cmd_evaluate,
    "graph": cmd_graph,
    "demo": cmd_demo,
}


def usage() -> None:
    print("\nResearch Paper Navigator")
    print("=" * 64)
    print("  python navigator.py <command> [options]\n")
    for name, fn in COMMANDS.items():
        first = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:<10} {first}")
    print("\n  Run `python check_env.py` first if anything fails.")
    print("  Options after the command are passed straight through to the")
    print("  underlying script, which prints its own --help.\n")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        usage()
        return
    cmd, *rest = sys.argv[1:]
    if cmd not in COMMANDS:
        print(f"\nUnknown command: {cmd}")
        usage()
        sys.exit(2)
    COMMANDS[cmd](rest)


if __name__ == "__main__":
    main()
