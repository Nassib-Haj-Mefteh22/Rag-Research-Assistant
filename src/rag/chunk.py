import json
import statistics
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import load_settings

DEFAULT_SIZE = 1000
DEFAULT_OVERLAP = 100


def pages_path() -> Path:
    return load_settings().processed_dir / "pages.jsonl"


def chunks_path(chunk_size: int, chunk_overlap: int) -> Path:
    return load_settings().processed_dir / f"chunks_{chunk_size}_{chunk_overlap}.jsonl"


def load_pages(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for page in pages:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def chunk_stats(chunks: list[dict], tiny: int = 100) -> dict:
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
