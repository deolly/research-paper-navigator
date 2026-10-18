# Checkpoints

The week-by-week record. Each folder holds the Python files as they stood at
that checkpoint and the completed worksheet submitted for it.

These are **snapshots, not the running system.** The code evolved across the
seven checkpoints, so the same filename differs between folders — this is the
point of keeping them. The consolidated, runnable version of the system is in
[`../src/`](../src/), and the raw logs and result files each checkpoint
produced are in [`../evidence/`](../evidence/).

| Checkpoint | What it added | What measuring it revealed |
|---|---|---|
| **1.1** Baseline | LLM answering with no retrieval at all | The control condition. Confident answers, invented citations. |
| **2.1** Retrieval | PDF loading, chunking, BM25, then dense vectors and 0.5/0.5 fusion | Keyword and vector scores are on incomparable scales; fusion needs normalising first. |
| **3.1** Evaluation | LLM-as-judge with a pinned judge model, stability runs | Score magnitudes are **not comparable across queries**, so tuning fusion weights on a 12-item set would fit noise. This decision holds for the rest of the project. |
| **4.1** Advanced retrieval | Paper graph: 137 nodes, 482 edges, and graph-aware retrieval | 16 of 153 PDFs yield no usable title, so they stay searchable but hold no edges. |
| **5.1** Agentic RAG | decide → act → reassess loop, four tools, answer / clarify / refuse | The same vague question clarifies 6 times in 10 and answers 4 — a 4.5× cost spread on one input. |
| **6.1** Security and cost | Five injection surfaces, per-role token metering, model ladder | Four bugs found in the **measuring instruments**, all biasing results toward looking safe. Parse failures were being scored as the system resisting. |
| **7.1** Production readiness | Input sanitizer, XML trust boundary, citation validator, fail-closed guard, router, cache | A fourth experimental arm proved the gains came from the prompt, not the code. The code never fired once in 1,050 trials. |

## Reading the evidence

Each checkpoint's raw output lives in the matching folder under
[`../evidence/`](../evidence/), and
[`../evidence/README.md`](../evidence/README.md) maps every claim in the
presentation to the file and line that supports it. The security statistics can
be recomputed from the raw logs with:

```bash
python evidence/summarize_security.py
```

## A note on duplication

Shared modules such as `capstone_retriever.py` appear in several checkpoint
folders. They are not identical copies — each is that week's version. Comparing
`checkpoint_5_1/capstone_agent.py` with `checkpoint_7_1/capstone_agent.py`
shows what changed when the safeguards went in, which is the kind of question
these folders exist to answer.
