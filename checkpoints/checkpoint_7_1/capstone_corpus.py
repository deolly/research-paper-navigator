"""Load and chunk the research-paper corpus."""
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_papers(papers_dir: str, limit: int | None = None) -> list[Document]:
    """One Document per PDF page, tagged with source paper and page number."""
    docs = []
    pdf_paths = sorted(Path(papers_dir).glob("*.pdf"))
    if limit:
        pdf_paths = pdf_paths[:limit]
        
    for path in pdf_paths:
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 50:          # skip blank or image-only pages
                continue
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": page_num},
            ))

    print(f"Loaded {len(docs)} pages from {len(pdf_paths)} PDFs")
    return docs


def chunk_papers(page_docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_docs)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}#p{c.metadata['page']}#c{i}"

    print(f"Split into {len(chunks)} chunks")
    return chunks