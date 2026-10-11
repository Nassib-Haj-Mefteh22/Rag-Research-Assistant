import re

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class Bm25Index:
    def __init__(self, chunks: list[dict]) -> None:
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def top(self, question: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]
