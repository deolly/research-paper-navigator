"""Test corpus-internal citations using space-insensitive title matching."""
import re
from pathlib import Path
from pypdf import PdfReader

PAPERS = Path(r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers")

def squash(s: str) -> str:
    """Lowercase and remove everything that isn't a letter or digit.
    Defeats PDF spacing artefacts entirely."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def is_proceedings(p1: str, n_pages: int) -> bool:
    low = p1.lower()
    markers = ("lecture notes in computer science", "editorial board",
               "commenced publication", "springer", "proceedings of the")
    hits = sum(m in low for m in markers)
    return hits >= 2 or n_pages > 150

def guess_title(p1: str) -> str:
    """Text before the abstract marker, capped."""
    t = p1
    for marker in ("Abstract", "ABSTRACT", "abstract"):
        i = t.find(marker)
        if 20 < i < 500:
            t = t[:i]
            break
    return t[:200]

pdfs = sorted(PAPERS.glob("*.pdf"))
print(f"Reading {len(pdfs)} PDFs...")

papers = {}     # name -> {"title_sq": ..., "reader": ...}
skipped = []
for p in pdfs:
    try:
        r = PdfReader(str(p))
    except Exception:
        continue
    p1 = " ".join((r.pages[0].extract_text() or "").split())
    if is_proceedings(p1, len(r.pages)):
        skipped.append((p.name, len(r.pages)))
        continue
    title = guess_title(p1)
    sq = squash(title)
    if len(sq) < 30:          # too short to match reliably
        continue
    papers[p.name] = {"title_sq": sq, "reader": r, "title": title}

print(f"  {len(papers)} papers kept, {len(skipped)} proceedings volumes skipped\n")
for n, pg in skipped:
    print(f"    skipped: {n}  ({pg} pages)")

# Test citation detection on 8 papers
print("\nCitation probe (space-insensitive title match):")
names = list(papers)[:8]
for name in names:
    r = papers[name]["reader"]
    start = int(len(r.pages) * 0.15)
    tail_sq = squash(" ".join(
        (pg.extract_text() or "") for pg in r.pages[start:]
    ))

    hits = []
    for other, info in papers.items():
        if other == name:
            continue
        probe = info["title_sq"][:45]       # distinctive 45-char window
        if probe and probe in tail_sq:
            hits.append(other)

    print(f"\n  {name[:55]:<57} -> {len(hits)} citation(s)")
    for h in hits[:6]:
        print(f"        {h}")