import json

from rag.config import ROOT
from rag.retrieve import DEFAULT_K, retrieve

DEFAULT_MODE = "hybrid+rerank"
METADATA_FILE = ROOT / "corpus" / "metadata.json"


def load_titles(path=METADATA_FILE) -> dict[str, str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["id"]: record["title"] for record in records}


def search_papers(
    query: str, k: int = DEFAULT_K, mode: str = DEFAULT_MODE, titles: dict[str, str] | None = None
) -> list[dict]:
    titles = load_titles() if titles is None else titles
    return [
        {
            "doc_id": chunk["doc_id"],
            "title": titles.get(chunk["doc_id"], "(title unknown)"),
            "page": chunk["page"],
            "score": chunk["score"],
            "text": chunk["text"],
        }
        for chunk in retrieve(query, k=k, mode=mode)
    ]
