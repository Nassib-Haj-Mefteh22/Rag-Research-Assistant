"""Step 7.3: keyword search (BM25) and Reciprocal Rank Fusion, to complement dense search.

WHY: dense embeddings capture meaning but blur exact tokens such as "141" or "RAGAs"
(step 7.2, Q9 and Q10). BM25 scores by exact word overlap, so it catches those.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60  # the constant from the original RRF paper; it damps the gap between top ranks


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. WHY keep digits: "141" must stay a searchable token."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse several ranked id lists: each id scores sum over lists of 1 / (k + rank), rank from 1.

    WHY ranks, not scores: cosine (0-1) and BM25 (0-30) are on different scales, so adding
    raw scores would let one method dominate. Ranks are comparable. Returns best first.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class Bm25Index:
    """BM25 over chunk texts; position i matches chunks[i], like the FAISS index."""

    def __init__(self, chunks: list[dict]) -> None:
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def top(self, question: str, n: int) -> list[int]:
        """Positions of the n best chunks, best first. Chunks with score 0 (no shared word) are
        dropped: a zero means "no evidence", and ranking them would just add noise to the fusion.
        """
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]
