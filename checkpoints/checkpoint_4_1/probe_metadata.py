"""Locate reference sections across all pages, and test email extraction."""
import re
from pathlib import Path
from pypdf import PdfReader

PAPERS = Path(r"C:\mit\rag\LAB_2\Capstone_Checkpoint_2_1_starter\ResearchPapers")

EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
BRACE = re.compile(r"\{([^}]+)\}@([\w.\-]+\.\w+)")

pdfs = sorted(PAPERS.glob("*.pdf"))[:6]

for path in pdfs:
    print("=" * 78)
    print(path.name)
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    p1 = " ".join((reader.pages[0].extract_text() or "").split())

    # Is this a proceedings volume rather than a paper?
    junk = ("lecture notes in computer science" in p1.lower()
            and "editorial board" in p1.lower())
    print(f"  pages={len(reader.pages)}  proceedings_volume={junk}")

    # Emails on page 1
    emails = set(EMAIL.findall(p1))
    for names, domain in BRACE.findall(p1):
        for n in names.split(","):
            emails.add(f"{n.strip()}@{domain}")
    print(f"  emails: {sorted(emails)}")

    # Find the references section anywhere in the document
    ref_page = None
    for i, pg in enumerate(reader.pages):
        t = (pg.extract_text() or "").lower()
        if re.search(r"\n\s*(references|bibliography)\s*\n", t) or \
           t.strip().startswith(("references", "bibliography")):
            ref_page = i
            break
    print(f"  references start page: {ref_page} of {len(reader.pages)}")

    if ref_page is not None:
        sample = " ".join((reader.pages[ref_page].extract_text() or "").split())
        print(f"  sample: {sample[:500]}")
    print()