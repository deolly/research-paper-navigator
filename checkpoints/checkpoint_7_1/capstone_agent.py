"""capstone_agent.py — agent-based RAG over the research-paper corpus.

The fixed pipelines built in Checkpoints 2.1 and 4.1 always do the same thing in
the same order. Checkpoint 4.1 measured why that is a problem: decomposition
gained 60 points on E3 (vocabulary gap) and lost 20 on E2 (the query already
matched the corpus wording), and graph traversal helped E4 while turning E3 into
an outright refusal. The conclusion there was that both techniques should be
applied CONDITIONALLY. Nothing in a fixed pipeline can make that choice.

This agent is the missing mechanism. At each step it sees what it has gathered
and picks one action:

    semantic_search        hybrid BM25 + vector retrieval (the 2.1 baseline)
    decompose_and_search   LLM rewrites the query into sub-queries first (4.1)
    follow_citation        walk `cites` edges out from a paper already gathered
    find_similar_papers    walk `similar_to` edges out from a paper already gathered
    clarify                ask the user a question and stop
    answer                 stop and answer from what has been gathered

find_by_author was considered and REJECTED: Checkpoint 4.1 found author emails
extractable from only 106 of 138 papers, and absent entirely from DreamCoder,
one of the two papers the E4 task depends on.

Fairness: however many chunks the agent gathers, exactly TOP_K are passed to the
answering model, capped at MAX_PER_PAPER per source. A win must come from
choosing better documents, not from being handed more of them.

SAFETY NOTE (observed, not hypothetical). An earlier version of this file forced
`answer` on the first malformed decide response and then answered with an empty
document list. On one --all run that produced a fluent reply citing SEVEN papers
that do not exist in this corpus, with quoted passages attributed to them. Two
guards were added as a result: decide() retries once before giving up, and
run_agent() refuses outright rather than answering when a parse failure has left
it with no retrieved context.

Usage
    python capstone_agent.py --task 1
    python capstone_agent.py --all
    python capstone_agent.py --query "..."
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import capstone_advanced as adv
from capstone_retriever import tokenize

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
MAX_STEPS = 4
TOP_K = adv.TOP_K          # 5 — identical to baseline and to the 4.1 system
MAX_PER_PAPER = 2          # diversity cap applied when pruning to TOP_K
SEARCH_K = 5               # chunks a single search action may add
GRAPH_FANOUT = 3           # papers a single graph action may pull in
DECIDE_RETRIES = 1         # retries on a malformed decide response

LOG_PATH = Path("checkpoint_5_1_agent.log")

DECIDE_SYSTEM = """You are a retrieval agent working over a corpus of 137 computer \
science research papers, indexed as 23,560 text chunks. The corpus is one research \
group's output on program synthesis, neurosymbolic methods and programming languages, \
spanning 2005-2025.

At each step you choose ONE action. Respond with ONLY a JSON object:
{"action": "<name>", "args": {...}, "reasoning": "<one sentence>"}

Available actions:

  semantic_search       args: {"query": "..."}
      Hybrid keyword + embedding retrieval. Use this first. Good when the query
      already uses the vocabulary a research paper would use.

  decompose_and_search  args: {"query": "..."}
      Rewrites the query into focused sub-queries using research terminology, then
      retrieves for each. Use this when a plain search returned low scores, or
      returned chunks scattered across unrelated papers, or when the question is
      phrased in everyday language that the papers are unlikely to use. It costs an
      extra model call, so do not use it when a plain search already worked.

  follow_citation       args: {"paper": "<filename already gathered>"}
      Pulls in papers the named paper cites, or that cite it. Use when you have one
      strong paper and the question asks how approaches compare, what earlier work
      something builds on, or how a line of research developed.

  find_similar_papers   args: {"paper": "<filename already gathered>"}
      Pulls in the most similar papers by embedding. Use when you need more papers on
      the same topic and citation links did not supply them.

  clarify               args: {"question": "..."}
      Ask the user one question and stop. Use ONLY when the request is IN SCOPE
      for this corpus but too broad or underspecified to retrieve for — for
      example naming a whole subfield when the corpus holds hundreds of papers on
      it. Do not use it for requests that fall outside the corpus entirely.

  answer                args: {}
      Stop and answer. Use as soon as the gathered chunks cover every part of the
      question. ALSO use it when the request is outside this corpus entirely —
      current prices, news, live data, anything not found in research papers. In
      that case answer by stating plainly that the corpus does not contain the
      information. Never ask the user a clarifying question about an out-of-scope
      request, and never offer information you have no tool to retrieve.

Guidance:
  - Never repeat an action with the same arguments.
  - Prefer answering early. Extra steps cost model calls and add irrelevant context.
  - A question that asks to compare, contrast, or name several approaches needs
    chunks from at least two different papers before you answer.
  - If two searches have returned nothing relevant, answer anyway and say the corpus
    does not appear to contain it. Do not keep searching."""

ANSWER_SYSTEM = (
    "You are answering questions about a corpus of computer science research papers. "
    "Use ONLY the provided documents, and quote from them where you can. Do not "
    "present work that is merely cited by these documents as though it were part of "
    "the corpus. If the documents do not contain the answer, say so rather than "
    "guessing."
)

NO_CONTEXT_REFUSAL = (
    "Retrieval did not complete for this question, so I have no documents to "
    "answer from and will not answer from memory. Please try again."
)


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — check your .env file.")
    return ChatOpenAI(model=adv.LLM_MODEL, temperature=temperature,
                      api_key=key, base_url=adv.OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 74}\n")


# ─── Agent state ─────────────────────────────────────────────────────
@dataclass
class Chunk:
    cid: str
    content: str
    score: float
    provenance: str


@dataclass
class AgentState:
    question: str
    chunks: dict[str, Chunk] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    llm_calls: int = 0
    steps: int = 0
    clarification: str | None = None
    parse_failures: int = 0
    refused_no_context: bool = False

    def papers(self) -> set[str]:
        return {c.cid.split("#")[0] for c in self.chunks.values()}

    def add(self, cid: str, content: str, score: float, prov: str) -> bool:
        if cid in self.chunks:
            return False
        self.chunks[cid] = Chunk(cid, content, score, prov)
        return True


# ─── Tools ───────────────────────────────────────────────────────────
def tool_semantic_search(state: AgentState, llm: ChatOpenAI, query: str) -> str:
    hits = adv.get_retriever().getTopK(query, SEARCH_K)
    added = sum(state.add(cid, txt, sc, f"semantic_search({query[:40]})")
                for cid, txt, sc in hits)
    top = hits[0][2] if hits else 0.0
    n_papers = len({c.split("#")[0] for c, _, _ in hits})
    return (f"added {added} new chunk(s) from {n_papers} paper(s); "
            f"top fused score {top:.3f}")


def tool_decompose_and_search(state: AgentState, llm: ChatOpenAI, query: str) -> str:
    subs, ok = adv.decompose_query(llm, query)
    state.llm_calls += 1
    if not ok:
        state.parse_failures += 1
        return "decomposition failed to parse; fell back to the raw query"
    hits = adv.merged_retrieve(subs, SEARCH_K)
    added = sum(state.add(cid, txt, sc, "decompose_and_search")
                for cid, txt, sc in hits)
    listed = " | ".join(s[:60] for s in subs)
    return f"sub-queries: {listed}\n      added {added} new chunk(s)"


def _best_chunk_from(paper: str, probe: str) -> tuple[str, str, float] | None:
    r = adv.get_retriever()
    idxs = adv._chunks_by_source.get(paper, [])
    if not idxs:
        return None
    scores = r._bm25.get_scores(tokenize(probe))
    best = max(idxs, key=lambda i: scores[i])
    doc = r._chunks[best]
    return doc.metadata["chunk_id"], doc.page_content, float(scores[best])


def _walk(state: AgentState, paper: str, edge_type: str, label: str) -> str:
    G, _ = adv.get_graph()
    if paper not in G:
        known = sorted(state.papers())[:3]
        return f"'{paper}' is not in the graph. Papers gathered so far: {known}"

    neighbours = [(nb, G.edges[paper, nb].get("weight", 1.0))
                  for nb in G.neighbors(paper)
                  if G.edges[paper, nb]["edge_type"] == edge_type
                  and nb not in state.papers()]
    if not neighbours:
        return f"no new {edge_type} neighbours for {paper.split('.')[0][:40]}"

    neighbours.sort(key=lambda kv: -kv[1])
    probe = (state.question + " " + " ".join(
        c.content for c in list(state.chunks.values())[:2]))[:1500]

    added, names = 0, []
    for nb, w in neighbours[:GRAPH_FANOUT]:
        got = _best_chunk_from(nb, probe)
        if got and state.add(got[0], got[1], w,
                             f"{label} from {paper.split('.')[0][:28]}"):
            added += 1
            names.append(nb.split(".")[0][:38])
    return f"added {added} chunk(s) via {edge_type}: {names}"


def tool_follow_citation(state: AgentState, llm: ChatOpenAI, paper: str) -> str:
    return _walk(state, paper, "cites", "cites")


def tool_find_similar(state: AgentState, llm: ChatOpenAI, paper: str) -> str:
    return _walk(state, paper, "similar_to", "similar_to")


TOOLS = {
    "semantic_search": (tool_semantic_search, "query"),
    "decompose_and_search": (tool_decompose_and_search, "query"),
    "follow_citation": (tool_follow_citation, "paper"),
    "find_similar_papers": (tool_find_similar, "paper"),
}


# ─── Decision step ───────────────────────────────────────────────────
def describe_state(state: AgentState) -> str:
    if not state.chunks:
        gathered = "(nothing gathered yet)"
    else:
        by_paper = defaultdict(list)
        for c in state.chunks.values():
            by_paper[c.cid.split("#")[0]].append(c)
        lines = []
        for paper, cs in sorted(by_paper.items(),
                                key=lambda kv: -max(c.score for c in kv[1])):
            best = max(cs, key=lambda c: c.score)
            lines.append(f"  {paper}  ({len(cs)} chunk(s), best score {best.score:.3f})")
            lines.append(f"      via {best.provenance}")
            lines.append(f"      \"{' '.join(best.content.split())[:180]}...\"")
        gathered = "\n".join(lines)

    hist = "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(state.history)) or "  (none)"
    return (f"Question: {state.question}\n\n"
            f"Actions taken so far:\n{hist}\n\n"
            f"Chunks gathered ({len(state.chunks)} from {len(state.papers())} paper(s)):\n"
            f"{gathered}")


def decide(state: AgentState, llm: ChatOpenAI,
           retries: int = DECIDE_RETRIES, verbose: bool = True) -> dict:
    """Ask the model for one action, retrying once on a malformed response.

    A parse failure is NOT the agent deciding it has enough information, so it is
    counted, printed, and retried rather than silently converted into `answer` —
    which is what the Checkpoint 5.1 starter did.
    """
    for attempt in range(retries + 1):
        raw = llm.invoke([
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=describe_state(state)),
        ]).content.strip()
        state.llm_calls += 1

        if raw.startswith("```"):
            raw = raw.strip("`")
            if "{" in raw:
                raw = raw[raw.find("{"):raw.rfind("}") + 1]

        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            state.parse_failures += 1
            if attempt < retries:
                if verbose:
                    print(f"    !! JSON parse failure, retrying "
                          f"({attempt + 1}/{retries})")
                continue
            return {"action": "answer", "args": {},
                    "reasoning": "!! JSON parse failure -> forced to answer"}

        action = d.get("action", "answer")
        if action not in TOOLS and action not in ("clarify", "answer"):
            state.parse_failures += 1
            if attempt < retries:
                if verbose:
                    print(f"    !! unknown action '{action}', retrying "
                          f"({attempt + 1}/{retries})")
                continue
            return {"action": "answer", "args": {},
                    "reasoning": f"!! unknown action '{action}' -> forced to answer"}

        return {"action": action, "args": d.get("args", {}) or {},
                "reasoning": d.get("reasoning", "")}


# ─── Final context: prune to TOP_K with a per-paper cap ──────────────
def select_context(state: AgentState) -> list[Chunk]:
    ranked = sorted(state.chunks.values(), key=lambda c: -c.score)
    out, per_paper = [], defaultdict(int)
    for c in ranked:
        paper = c.cid.split("#")[0]
        if per_paper[paper] >= MAX_PER_PAPER:
            continue
        per_paper[paper] += 1
        out.append(c)
        if len(out) >= TOP_K:
            break
    # Top up if the diversity cap left us short
    if len(out) < TOP_K:
        chosen = {c.cid for c in out}
        for c in ranked:
            if c.cid not in chosen:
                out.append(c)
                chosen.add(c.cid)
            if len(out) >= TOP_K:
                break
    return out


def answer(state: AgentState, llm: ChatOpenAI, selected: list[Chunk]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c.cid}]  (retrieved via {c.provenance})\n{c.content}" for c in selected)
    state.llm_calls += 1
    return llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {state.question}"),
    ]).content


# ─── The loop ────────────────────────────────────────────────────────
def run_agent(question: str, llm: ChatOpenAI, verbose: bool = True) -> dict:
    state = AgentState(question=question)

    for step in range(1, MAX_STEPS + 1):
        state.steps = step
        d = decide(state, llm, verbose=verbose)
        action, args = d["action"], d["args"]

        if verbose:
            print(f"\n  STEP {step}: {action}")
            print(f"    reasoning: {d['reasoning']}")

        if action == "answer":
            state.history.append(f"answer — {d['reasoning'][:60]}")
            break

        if action == "clarify":
            q = args.get("question", "Could you be more specific?")
            state.clarification = q
            state.history.append(f"clarify — {q[:60]}")
            if verbose:
                print(f"    ASKS USER: {q}")
            break

        fn, argname = TOOLS[action]
        value = args.get(argname, question if argname == "query" else "")
        result = fn(state, llm, value)
        state.history.append(f"{action}({str(value)[:50]}) -> {result[:80]}")
        if verbose:
            print(f"    {argname}: {value}")
            print(f"    result: {result}")
    else:
        state.history.append(f"stopped at MAX_STEPS={MAX_STEPS}")
        if verbose:
            print(f"\n  step budget exhausted ({MAX_STEPS})")

    if state.clarification:
        final = f"[CLARIFICATION REQUESTED] {state.clarification}"
        selected = []
    elif not state.chunks and state.parse_failures:
        # A forced answer with no retrieved context must NOT be answered from the
        # model's own knowledge. Observed once on a --all run: a parse failure
        # produced a reply citing seven papers and quoting passages from them,
        # none of which exist in this corpus.
        state.refused_no_context = True
        state.history.append("refused — no context after parse failure")
        selected = []
        final = NO_CONTEXT_REFUSAL
        if verbose:
            print("\n  !! REFUSED: parse failure left no retrieved context; "
                  "answering from model knowledge is not permitted.")
    else:
        selected = select_context(state)
        final = answer(state, llm, selected)

    if verbose:
        print(f"\n  CONTEXT SENT TO MODEL ({len(selected)} chunks, "
              f"{len({c.cid.split('#')[0] for c in selected})} distinct papers):")
        for c in selected:
            print(f"      {c.score:7.3f}  {c.cid}")
            print(f"               via {c.provenance}")
        print(f"\n  COST: {state.steps} step(s), {state.llm_calls} LLM call(s)"
              + (f", {state.parse_failures} parse failure(s)"
                 if state.parse_failures else ""))
        print(f"\n  ANSWER:\n{final}\n")

    log(f"AGENT | {question}",
        "history:\n" + "\n".join(f"  {h}" for h in state.history)
        + f"\nselected={[c.cid for c in selected]}"
        + f"\nsteps={state.steps} llm_calls={state.llm_calls}"
        + f"\nparse_failures={state.parse_failures}"
        + f"\nrefused_no_context={state.refused_no_context}"
        + f"\nanswer={final}")

    return {
        "question": question,
        "answer": final,
        "steps": state.steps,
        "llm_calls": state.llm_calls,
        "chunks_gathered": len(state.chunks),
        "chunks_used": len(selected),
        "distinct_papers": len({c.cid.split("#")[0] for c in selected}),
        "history": state.history,
        "clarified": state.clarification is not None,
        "parse_failures": state.parse_failures,
        "refused_no_context": state.refused_no_context,
    }


# ─── Test tasks ──────────────────────────────────────────────────────
TEST_TASKS = [
    # 1. Cross-paper comparison. One search cannot cover both halves, and the two
    #    halves use different vocabulary.
    "How does the sketching approach to program synthesis differ from library "
    "learning approaches like DreamCoder?",

    # 2. Lineage. Requires finding one paper, then traversing citations out of it.
    "What earlier work does the Stitch library-learning paper build on, and what "
    "did it change?",

    # 3. Ambiguous. A fixed pipeline must guess; the agent can ask.
    "Tell me about the synthesis work in this corpus.",

    # 4. The 4.1 vocabulary-gap item, to see whether the agent reaches for
    #    decompose_and_search on its own.
    "How can a programmer leave gaps in their code for the computer to fill in "
    "automatically?",

    # 5. Out-of-corpus control. The agent should not search indefinitely.
    "What is the current market capitalization of Nvidia?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, help="run one test task by number (1-5)")
    ap.add_argument("--all", action="store_true", help="run all test tasks")
    ap.add_argument("--query", help="run an arbitrary question")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each question N times (for reliability measurement)")
    args = ap.parse_args()

    llm = make_llm()

    if args.query:
        questions = [args.query]
    elif args.all:
        questions = TEST_TASKS
    elif args.task:
        questions = [TEST_TASKS[args.task - 1]]
    else:
        questions = [TEST_TASKS[0]]

    results = []
    for q in questions:
        for run in range(args.repeat):
            print("\n" + "=" * 78)
            label = f"TASK: {q}"
            if args.repeat > 1:
                label += f"   [run {run + 1}/{args.repeat}]"
            print(label)
            print("=" * 78)
            results.append(run_agent(q, llm))

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("  COST SUMMARY")
        print("=" * 78)
        print(f"  {'run':<5} {'steps':<7} {'calls':<7} {'gathered':<10} "
              f"{'used':<6} {'papers':<8} {'parse_fail':<12} status")
        for i, r in enumerate(results, 1):
            status = ("clarified" if r["clarified"]
                      else "REFUSED (no context)" if r["refused_no_context"]
                      else "")
            print(f"  {i:<5} {r['steps']:<7} {r['llm_calls']:<7} "
                  f"{r['chunks_gathered']:<10} {r['chunks_used']:<6} "
                  f"{r['distinct_papers']:<8} {r['parse_failures']:<12} {status}")

        n = len(results)
        avg_calls = sum(r["llm_calls"] for r in results) / n
        n_parse = sum(1 for r in results if r["parse_failures"])
        n_refused = sum(1 for r in results if r["refused_no_context"])
        print(f"\n  Mean LLM calls per run: {avg_calls:.1f}"
              f"   (baseline pipeline = 1, Checkpoint 4.1 pipeline = 2)")
        print(f"  Runs with >=1 parse failure: {n_parse}/{n}")
        print(f"  Runs refused for lack of context: {n_refused}/{n}")

    print(f"\nLogged to {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()