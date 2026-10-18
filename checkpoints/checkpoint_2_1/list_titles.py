from capstone_corpus import load_papers

docs = load_papers(r".\ResearchPapers")

seen = set()
lines = []
for d in docs:
    src = d.metadata["source"]
    if src in seen or d.metadata["page"] != 1:
        continue
    seen.add(src)
    head = " ".join(d.page_content.split())[:110]
    lines.append(f"{src[:55]:<57} | {head}")

with open("titles.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print(f"Wrote {len(lines)} titles to titles.txt")