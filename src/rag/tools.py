"""Step 12.3: search_papers(query) -> simple dicts, the one function other code should call.

WHY a wrapper: retrieve() also takes chunking parameters that a caller should not choose, and its
dicts carry no paper title. This hides the first and adds the second, so a demo, notebook or agent
gets readable, JSON-friendly results without knowing how the index is built.
"""

import json

from rag.config import ROOT
from rag.retrieve import DEFAULT_K, retrieve

DEFAULT_MODE = "hybrid+rerank"  # step 11.7: best Hit@4 (0.857) and MRR (0.676), fewest refusals
METADATA_FILE = ROOT / "corpus" / "metadata.json"


def load_titles(path=METADATA_FILE) -> dict[str, str]:
    """doc_id -> paper title, so sources are readable instead of bare arXiv ids."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["id"]: record["title"] for record in records}


def search_papers(
    query: str, k: int = DEFAULT_K, mode: str = DEFAULT_MODE, titles: dict[str, str] | None = None
) -> list[dict]:
    """Top-k passages for a query, best first: doc_id, title, page, score, text.

    `titles` (doc_id -> title) can be passed in to avoid re-reading metadata.json on every call.
    Scores are only comparable within one call and one mode.
    """
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
