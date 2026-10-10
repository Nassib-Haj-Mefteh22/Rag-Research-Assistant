"""Step 4: split pages into chunks with metadata (doc_id, page, chunk_id, text).

WHY chunks: a page mixes several topics, so its embedding is a blurry average and it wastes
prompt tokens. A chunk of a few hundred tokens stays close to one idea.
"""

import json
import statistics
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import load_settings

DEFAULT_SIZE = 1000  # characters (~250 tokens, well under bge-small's 512-token limit)
DEFAULT_OVERLAP = 100  # 10% of the size, so size is the only variable between experiments


def pages_path() -> Path:
    return load_settings().processed_dir / "pages.jsonl"


def chunks_path(chunk_size: int, chunk_overlap: int) -> Path:
    """One file per config, so experiments never overwrite each other (step 5 aligns on it)."""
    return load_settings().processed_dir / f"chunks_{chunk_size}_{chunk_overlap}.jsonl"


def load_pages(path: Path) -> list[dict]:
    """Read pages.jsonl: one {doc_id, page, text} per line."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Split each page on its own and tag every chunk with its (doc_id, page).

    WHY per page: gold labels and citations are (doc_id, page); splitting a whole document
    would let chunks straddle pages and lose the exact page number.
    WHY recursive: it cuts at paragraph, then line, then sentence, then word boundaries,
    and only cuts mid-word as a last resort.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for page in pages:
        # The splitter can return whitespace-only pieces; they carry no information.
        texts = [t for t in splitter.split_text(page["text"]) if t.strip()]
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "chunk_id": f"{page['doc_id']}_p{page['page']}_c{i}",
                    "doc_id": page["doc_id"],
                    "page": page["page"],
                    "text": text,
                }
            )
    return chunks


def write_chunks(chunks: list[dict], path: Path) -> None:
    """One JSON object per line (same format as pages.jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def chunk_stats(chunks: list[dict], tiny: int = 100) -> dict:
    """Length summary in characters.

    WHY: checks the splitter did what chunk_size says (max <= size, mean a bit below it) and
    counts chunks shorter than `tiny`, which embed poorly (little text, often a stray heading).
    """
    lengths = [len(c["text"]) for c in chunks]
    return {
        "count": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "tiny": sum(1 for n in lengths if n < tiny),
    }


if __name__ == "__main__":
    pages = load_pages(pages_path())
    chunks = chunk_pages(pages, DEFAULT_SIZE, DEFAULT_OVERLAP)
    out = chunks_path(DEFAULT_SIZE, DEFAULT_OVERLAP)
    write_chunks(chunks, out)
    print(f"pages: {len(pages)} -> chunks: {len(chunks)}")
    print(f"size={DEFAULT_SIZE}, overlap={DEFAULT_OVERLAP}")
    print(f"wrote {out}")
    stats = chunk_stats(chunks)
    print(
        f"chars: mean {stats['mean']:.0f} | median {stats['median']:.0f} | "
        f"min {stats['min']} | max {stats['max']} | under 100 chars: {stats['tiny']}"
    )
    for c in chunks[:3]:
        print(f"  {c['chunk_id']} ({len(c['text'])} chars): {c['text'][:70]!r}")
