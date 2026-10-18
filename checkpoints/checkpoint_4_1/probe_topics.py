"""Test keyword-based topic extraction and shares_topic edge density."""
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from pypdf import PdfReader

PAPERS = Path(r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers")

STOP = set("""the a an and or but of to in on at by for with from is are was were be been
being this that these those we our it its as if not no can may their they he she
paper we present show propose introduce approach method results using used use
which such also more most than then there here into over under between each both
all any some other new work works section figure table et al arxiv preprint
conference proceedings international journal pages vol pp acm ieee springer""".split())

def keywords(text: str, n: int = 12) -> list[str]:
    words = re.findall(r"[a-z][a-z\-]{3,}", text.lower())
    words = [w for w in words if w not in STOP]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    counts = Counter(bigrams)
    return [w for w, c in counts.most_common(n) if c >= 3]

def is_proceedings(p1, n):
    low = p1.lower()
    m = ("lecture notes in computer science", "editorial board",
         "commenced publication", "springer", "proceedings of the")
    return sum(x in low for x in m) >= 2 or n > 150

topics = {}
for p in sorted(PAPERS.glob("*.pdf")):
    try:
        r = PdfReader(str(p))
    except Exception:
        continue
    p1 = " ".join((r.pages[0].extract_text() or "").split())
    if is_proceedings(p1, len(r.pages)):
        continue
    # Use first 3 pages: title, abstract, intro — where topic words concentrate
    head = " ".join((pg.extract_text() or "") for pg in r.pages[:3])
    kw = keywords(head)
    if kw:
        topics[p.name] = set(kw)

print(f"{len(topics)} papers with extracted keywords\n")

for name in list(topics)[:5]:
    print(f"  {name[:50]:<52} {sorted(topics[name])[:6]}")

# Edge density at different thresholds
print("\nshares_topic edge counts by threshold:")
for thresh in (1, 2, 3, 4):
    edges = sum(1 for a, b in combinations(topics, 2)
                if len(topics[a] & topics[b]) >= thresh)
    avg = 2 * edges / len(topics) if topics else 0
    print(f"  >= {thresh} shared keyword(s): {edges:6,} edges   (avg degree {avg:.1f})")

# Check the two papers E4 needs connected
a = "DBLP_journals_pacmpl_BowersOWGTES23.pdf"
b = "DBLP_journals_corr_abs-2006-08381.pdf"
if a in topics and b in topics:
    shared = topics[a] & topics[b]
    print(f"\nStitch <-> DreamCoder shared keywords: {len(shared)}")
    print(f"  {sorted(shared)}")
else:
    print(f"\nOne of the E4 papers missing: Stitch={a in topics} DreamCoder={b in topics}")