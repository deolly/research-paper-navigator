# Research Paper Navigator

A retrieval-augmented question-answering system over a corpus of 153 computer
science research papers on program synthesis and library learning. It answers
questions about what the papers say, cites the paper each claim came from, and
refuses when the corpus does not contain an answer.

Built incrementally across seven checkpoints of the MIT Professional Education
*Agentic RAG* program. The engineering interest is less in the retrieval stack, 
which is conventional than in what happened when it was measured: a prompt
injection that the "hardened" prompt made *worse*, a denial-of-service that
crashed the planner in 30% of trials, and four bugs found in the measuring
instruments themselves. See [`evidence/README.md`](evidence/README.md).

---

## The problem

A researcher with a folder of 153 papers has three kinds of question that
keyword search answers badly:

| Question | Why search fails |
|---|---|
| "What is a sketch in program synthesis?" | Fine - one paper, one definition. Search handles this. |
| "How does sketching differ from library learning like DreamCoder?" | The answer exists in no single paper. It needs two retrievals and a synthesis. |
| "What earlier work does Stitch build on, and what did it change?" | Needs the citation relationships between papers, which no amount of text search recovers. |

The system routes each question to the cheapest path that can answer it, and
grounds every answer in retrieved text rather than in the model's own knowledge.

## What it does not do

It is not a chatbot and not a search engine. It will not answer from general
knowledge: asked for Nvidia's market capitalisation, it refuses, because that
is not in the corpus. It has no web access, no user accounts, and no persistent
conversation state.

---

## Quick start

**Requirements:** Python 3.10 or newer (tested on 3.12.10), an
[OpenRouter](https://openrouter.ai/keys) API key, and the corpus PDFs — see
[Corpus](#corpus) below.

```bash
git clone <this repository>
cd research-paper-navigator

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

copy .env.example .env         # Windows   (cp on macOS/Linux)
# then edit .env: add OPENROUTER_API_KEY and set CORPUS_DIR

python check_env.py
```

`check_env.py` verifies the runtime, every dependency, every module, your key
and the corpus, and tells you what to fix if something is missing. It makes no
API calls and costs nothing. Run it before anything else.

Then, free and offline:

```bash
python navigator.py selftest
```

That runs the four regression suites — input sanitizer, citation validator,
query router, cache separability — in about two seconds with no API calls. If it
passes, the system is correctly installed.

To ask a real question (this one costs a few cents):

```bash
python navigator.py ask "What is the sketching approach to program synthesis?"
```

### Configuration

Everything lives in `.env`. There are only two variables:

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | **yes** | Every LLM call and the embeddings |
| `CORPUS_DIR` | no | Absolute path to the PDFs. Defaults to `./ResearchPapers` |
| `CHROMA_DIR` | no | Where the vector index is persisted. Defaults to `./chroma_papers` |

### First run builds the index

The first command that touches the corpus embeds roughly 23,560 chunks. **This
takes about ten minutes and costs money.** The index is then cached in
`chroma_papers/` and reused.

The path is anchored to the repository root, not to your working directory.
That was a real bug: running the same code from a different folder silently
re-embedded the whole corpus.

---

## Commands

```
python navigator.py                     list the commands
python navigator.py selftest            offline regression tests -- free
python navigator.py ask "..."           answer one question
python navigator.py security            injection probes, four arms
python navigator.py cost                token accounting
python navigator.py audit               the checkpoint 6.1 security audit
python navigator.py evaluate            retrieval and answer quality
python navigator.py graph               build or inspect the paper graph
python navigator.py demo                print the presentation run order
```

`navigator.py` adds no behaviour. It sets `sys.argv`, calls the underlying
module's own `main()`, and prints the equivalent raw command, so anything it
does can be reproduced by hand and the modules stay independently runnable.

Options after the command pass straight through - `python navigator.py security
--repeat 10 --model openai/gpt-5.4-mini` works, and each underlying script has
its own `--help`.

---

## How it works

![Architecture: request path and subsystems](docs/architecture.png)

### Retrieval

Pages are extracted with `pypdf`, split at 1,000 characters with 150 overlap,
and indexed two ways: BM25 Okapi over tokens, and dense vectors from
`text-embedding-3-small` in Chroma. Both return 20 candidates, scores are
min-max normalised to make them comparable, and fused at equal weight. A
diversity cap of 2 chunks per paper stops one verbose paper from filling the
context window.

Equal weighting is deliberate and not tuned. Checkpoint 3.1 established that
score magnitudes in this system are not comparable across queries, so tuning
weights against a 12-item evaluation set would have fitted noise.

### The paper graph

Built once and cached. From 153 PDFs, 137 yield usable titles - the other 16 are
scanned or have mangled text - producing 482 edges:

| Edge type | Count | Built from |
|---|---|---|
| `similar_to` | 382 | Title and abstract similarity |
| `cites` | 73 | Reference-list matching against corpus titles |
| `duplicate_of` | 27 | Near-identical papers (preprint and published versions) |

The graph is what makes lineage questions answerable. `follow_citation` and
`find_similar_papers` traverse it.

### The agentic loop

At each step the planner sees the question and what has been gathered, then
emits one action: a tool call, `clarify`, or `answer`. Capped at 4 steps and 5
chunks. The planner is the expensive component - **62% of all tokens** - and
the one measurement that mattered most, because it is also the component that
crashes under attack.

### Models

Configured through OpenRouter, defaulting to `openai/gpt-5.4-mini`. The ladder
spans `gpt-5.4-nano` (cheap) through `gpt-5.4` (expensive). The LLM judge used
in evaluation is **pinned separately** so a model under test cannot grade its
own output.

`qwen/qwen3-8b` appears in the ladder and in the checkpoint 6.1 results. It was
deprecated on 2026-10-09 and now returns HTTP 404. Those results cannot be
reproduced.

---

## Results

Full detail, with every p-value recomputable from the raw logs, in
[`evidence/README.md`](evidence/README.md).

![Security results: five attack surfaces across four experimental arms](docs/security.png)

| Measure | Before | After | |
|---|---|---|---|
| Fake-source acceptance (`gpt-5.4-nano`) | 20% [11.8–31.8] | 0% [0–11.4] | p = 0.0071 |
| Planner parse failure under poison | 30% [19.9–42.5] | 0% [0-6.0] | p = 0.0000017 |
| Sanitizer false positives on benign queries | 4/12 | 0/18 | |
| Citation validator against labelled cases | - | 7/7 | |
| Query routing accuracy | 8/12 = 67% | 11/12 = 92% | |
| Tokens: direct path vs agentic path | - | 1,381 vs 3,312 | 58% saving |

### What the measurements do not show

Two results are worth stating because they are the honest ones:

**The defences are prompts, not code.** The citation validator withheld 0
answers and the fail-closed guard forced 0 refusals across 1,050 trials. Both
pass their own regression tests; neither has ever fired in a live run. A fourth
experimental arm (`xml-only`, prompt changes with the code disabled) was added
specifically to test whether the code was contributing, and it is not.

**The hardened prompt made one attack worse.** On the fake-source injection,
checkpoint 6.1's hardened prompt scored 20% against baseline's 15%. The attack
was closed only by labelling where each piece of text came from. Instructing a
model to "answer only from the retrieved context" is exactly what the attack
exploits, so more emphasis does not help.

---

## Known limitations

For reviewers, in the order I would want them known:

1. **The security defences live in a system prompt.** A model swap, a
   provider-side update or a temperature change can undo them and nothing in the
   code would notice. The deterministic safeguards that should be the safety net
   are unproven in production.
2. **Five attacks, all written by the author.** Zero failures against an attack
   I invented means my imagination held, not that the system did.
3. **One provider, one corpus, two models.** Every number comes from two cheap
   OpenAI models via OpenRouter against 153 papers in one narrow field.
4. **Both vulnerabilities are specific to the cheapest model.** `gpt-5.4-mini`
   shows 0 compromises on every probe. That is the central tension: the cheap
   model is where the savings are and where the failures are.
5. **Behavior is non-deterministic, including correct behavior.** The same
   vague question produces a clarifying question 6 times in 10 and an answer the
   other 4, a 4.5x spread in cost for one input. "0 failures in 30 trials"
   means the true rate is below 11%, not zero.
6. **The cost saving has no denominator.** A simple query costs 58% less on the
   direct path. What fraction of real queries are simple was never measured, so
   the headline figure is an upper bound on an unknown quantity.
7. **No semantic cache.** The lexical cache was removed after measurement: it
   scored "Does Stitch outperform DreamCoder?" and "Does DreamCoder outperform
   Stitch?" as identical. Exact match only, so the real-world hit rate is low.
8. **No monitoring.** No alerting on parse-failure rate, no refusal-rate
   tracking. Every number in this repository came from running a script by hand.
9. **16 of 153 papers are missing from the graph.** Their titles could not be
   extracted from the PDF text. They remain searchable but have no citation or
   similarity edges.

---

## Corpus

The 153 PDFs are **not in this repository**. They are third-party published
research papers, roughly 400 MB, and not mine to redistribute.
[`evidence/corpus_titles.txt`](evidence/corpus_titles.txt) lists every title so
the corpus is identifiable.

To run the system, point `CORPUS_DIR` in `.env` at any folder of PDFs. It works
with a different corpus, results in `evidence/` will not reproduce, but the
offline `selftest` suite still passes, since it tests the safeguards rather than
the content.

The vector index (`chroma_papers/`, 347 MB) and paper graph (`graph_cache/`,
13 MB) are also excluded. Both rebuild from the PDFs on first run.

---

## Repository layout

The repository serves two readers, so it is organised two ways.

**By checkpoint**, for anyone following the course: `checkpoints/` holds the
Python files and completed worksheet for each of the seven checkpoints, as they
stood that week.

**By function**, for anyone running the system: `src/` holds the consolidated,
current code, and `evidence/` holds the measurements with a map from each claim
to the file that supports it.

```
├── navigator.py              single entry point -- start here
├── check_env.py              preflight check; run this first
├── requirements.txt          pinned to the versions the results were measured on
├── .env.example              configuration template
│
├── checkpoints/              WEEK BY WEEK -- code and worksheet per checkpoint
│   ├── README.md             what each checkpoint added, and what it revealed
│   ├── checkpoint_1_1/ ... checkpoint_7_1/
│
├── src/                      THE RUNNING SYSTEM -- consolidated current code
│   ├── paths.py              resolves CORPUS_DIR and CHROMA_DIR
│   ├── capstone_corpus.py    PDF loading, chunking
│   ├── capstone_retriever.py hybrid BM25 + dense retrieval
│   ├── capstone_graph.py     paper graph construction
│   ├── capstone_advanced.py  advanced retrieval (checkpoint 4.1)
│   ├── capstone_agent.py     agentic decide/act/reassess loop (5.1)
│   ├── capstone_security_cost.py  injection probes, token metering (6.1)
│   ├── capstone_production.py     sanitizer, trust boundary, validator,
│   │                              router, cache, four-arm harness (7.1)
│   ├── capstone_eval_3_1.py  LLM-as-judge evaluation
│   ├── capstone_eval_4_1.py  advanced-retrieval evaluation
│   └── inspect_probes.py     probe-result inspector
│
├── evidence/                 THE MEASUREMENTS
│   ├── README.md             claim-to-file map -- read this for results
│   ├── summarize_security.py recomputes every rate and p-value from the logs
│   ├── corpus_titles.txt     the 153 paper titles
│   └── checkpoint_{1_1..7_1}/  raw logs and result files per checkpoint
│
└── docs/
    ├── architecture.png / .svg   request path and subsystems
    ├── security.png / .svg       attack surfaces and the four-arm result
    └── Capstone_Presentation_Adeola_Adeoti.pptx
```

The same filename appears in several checkpoint folders and again in `src/`.
Those are not duplicates: each checkpoint folder holds that week's version, and
diffing them is how the evolution of the system is read. `src/` is the only
copy that runs.

Module filenames keep their checkpoint-era names. Renaming them would have meant
rewriting imports across 84 KB of working, measured code for no functional gain,
and the names tie each module to the evidence folder that tests it.


## Dependencies

Pinned in `requirements.txt` to the exact versions every result was measured
with. `langchain-core` 1.6.0, `langchain-openai` 1.6.0, `langchain-chroma`
1.1.0, `chromadb` 1.5.9, `rank-bm25` 0.2.2, `networkx` 3.6.1, `numpy` 2.5.2,
`pypdf` 6.16.2, `python-dotenv` 1.2.3.

Pinned rather than loose because an unpinned install in six months is a
different system, and the numbers in `evidence/` would no longer describe it.


---

## AI assistance disclosure

AI assistance was used in this capstone. It computed the reported statistics, 
drafted the README, the evidence documentation, and the presentation slide diagrams.

The author directed the work throughout: defining scope at each checkpoint,
executing every run, supplying the corpus and course context, and deciding what
was kept, revised or discarded. Several key decisions were the author's, including
the addition of the fourth experimental arm that isolated prompt effects from code
effects and the result that reframed the security findings.

All results reported here come from runs the author executed. Where AI-generated
analysis conflicted with the recorded data, the data was followed: four measurement
bugs and three self-authored test sets were identified and corrected during the
project, and several earlier claims were revised after re-reading the raw logs. The
author reviewed the code, the measurements and the conclusions, and is responsible
for the accuracy of what is submitted.
