"""Step 3: PDF -> cleaned (doc_id, page, text) records in pages.jsonl."""

import json
from pathlib import Path

import pymupdf

from rag.clean import clean_page, drop_references, remove_headers_footers
from rag.config import ROOT

RAW_DIR = ROOT / "data" / "raw"
PAGES_FILE = ROOT / "data" / "processed" / "pages.jsonl"


def parse_pdf(path: Path) -> list[dict]:
    """Extract one record {doc_id, page, text} per page of a single PDF.

    WHY per page: gold labels and citations are (doc_id, page), so the page number
    must travel with the text from the very first step. No cleaning here on purpose:
    when something looks wrong in 3.2 we can tell if the PDF or our rules caused it.
    """
    doc_id = path.stem  # "2608.21558.pdf" -> "2608.21558", same id as corpus_ids.txt
    records: list[dict] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            records.append(
                {
                    "doc_id": doc_id,
                    # pymupdf counts pages from 0; PDF viewers and citations from 1.
                    "page": index + 1,
                    "text": page.get_text(),
                }
            )
    return records


def parse_corpus(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Parse every PDF in raw_dir. Sorted so the output order is deterministic."""
    records: list[dict] = []
    for path in sorted(raw_dir.glob("*.pdf")):
        records.extend(parse_pdf(path))
    return records


def check_page_counts(raw_dir: Path = RAW_DIR) -> list[str]:
    """Return a message per PDF whose record count differs from pymupdf's page count."""
    problems: list[str] = []
    for path in sorted(raw_dir.glob("*.pdf")):
        with pymupdf.open(path) as doc:
            expected = doc.page_count
        got = len(parse_pdf(path))
        if got != expected:
            problems.append(f"{path.stem}: {got} records vs {expected} pages")
    return problems


def process_doc(page_texts: list[str]) -> list[tuple[int, str]]:
    """Run the full cleaning chain on ONE document; return (page_number, text) pairs.

    Order is fixed by what we learned in 3.3/3.4: headers/footers and references need the
    RAW line structure, so they run before clean_page joins the lines. Pages that end up
    empty (a lone page number, a references-only page) are dropped. Page numbers stay the
    ORIGINAL PDF numbers, so a dropped page leaves a gap: gold labels use real page numbers.
    """
    texts = remove_headers_footers(page_texts)
    texts = drop_references(texts)
    cleaned = [(i + 1, clean_page(text)) for i, text in enumerate(texts)]
    return [(page, text) for page, text in cleaned if text]


def build_pages(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Parse and clean every PDF; one {doc_id, page, text} record per kept page."""
    records: list[dict] = []
    for path in sorted(raw_dir.glob("*.pdf")):
        raw = parse_pdf(path)
        for page, text in process_doc([rec["text"] for rec in raw]):
            records.append({"doc_id": path.stem, "page": page, "text": text})
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """One JSON object per line. ensure_ascii=False keeps accents/Greek letters readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raw_pages = parse_corpus()
    records = build_pages()
    write_jsonl(records, PAGES_FILE)
    print(f"PDFs: {len(list(RAW_DIR.glob('*.pdf')))} | raw pages: {len(raw_pages)}")
    print(f"kept pages: {len(records)} | dropped: {len(raw_pages) - len(records)}")
    raw_chars = sum(len(r["text"]) for r in raw_pages)
    clean_chars = sum(len(r["text"]) for r in records)
    print(f"chars: {raw_chars} raw -> {clean_chars} clean")
    print(f"wrote {PAGES_FILE}")
    for rec in records[:5]:
        print(f"  {rec['doc_id']} p.{rec['page']}: {rec['text'][:80]!r}")
