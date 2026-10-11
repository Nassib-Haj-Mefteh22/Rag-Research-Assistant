from collections.abc import Callable
from functools import cache

import faiss
import numpy as np

from rag.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, chunks_path, load_pages
from rag.embed import embed_query, load_embedder
from rag.hybrid import Bm25Index, rrf
from rag.index import index_path, load_index, search
from rag.rerank import Reranker

DEFAULT_K = 4
HYBRID_POOL = 20


class Retriever:
    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict],
        embed: Callable[[str], np.ndarray],
        reranker: Reranker | None = None,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embed = embed
        self._bm25: Bm25Index | None = None
        self.reranker = reranker or Reranker()

    @classmethod
    def from_config(cls, chunk_size: int = DEFAULT_SIZE, chunk_overlap: int = DEFAULT_OVERLAP):
        chunks = load_pages(chunks_path(chunk_size, chunk_overlap))
        index = load_index(index_path(chunk_size, chunk_overlap), chunks)
        model = load_embedder()
        return cls(index, chunks, lambda question: embed_query(model, question))

    def retrieve(self, question: str, k: int = DEFAULT_K, mode: str = "dense") -> list[dict]:
        if not question.strip():
            raise ValueError("question is empty")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        base, _, rerank = mode.partition("+")
        if base not in ("dense", "hybrid") or rerank not in ("", "rerank"):
            raise ValueError(
                f"mode must be dense, hybrid, dense+rerank or hybrid+rerank, got {mode!r}"
            )
        pool = HYBRID_POOL if rerank else k
        if base == "dense":
            hits = search(self.index, self.chunks, self.embed(question), pool)
        else:
            hits = self._hybrid(question, pool)
        if rerank:
            hits = self._rerank(question, hits)[:k]
        return [
            {**chunk, "rank": rank, "score": score}
            for rank, (score, chunk) in enumerate(hits, 1)
        ]

    def _rerank(self, question: str, hits: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        scores = self.reranker.score(question, [chunk["text"] for _, chunk in hits])
        ranked = sorted(zip(scores, (chunk for _, chunk in hits), strict=True),
                        key=lambda pair: pair[0], reverse=True)
        return [(score, chunk) for score, chunk in ranked]

    def _hybrid(self, question: str, k: int) -> list[tuple[float, dict]]:
        if self._bm25 is None:
            self._bm25 = Bm25Index(self.chunks)
        dense = search(self.index, self.chunks, self.embed(question), HYBRID_POOL)
        dense_ids = [chunk["chunk_id"] for _, chunk in dense]
        bm25_ids = [self.chunks[i]["chunk_id"] for i in self._bm25.top(question, HYBRID_POOL)]
        by_id = {c["chunk_id"]: c for c in self.chunks}
        fused = rrf([dense_ids, bm25_ids])[:k]
        return [(score, by_id[chunk_id]) for chunk_id, score in fused]


@cache
def _default_retriever(chunk_size: int, chunk_overlap: int) -> Retriever:
    return Retriever.from_config(chunk_size, chunk_overlap)


def retrieve(
    question: str,
    k: int = DEFAULT_K,
    chunk_size: int = DEFAULT_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP,
    mode: str = "dense",
) -> list[dict]:
    return _default_retriever(chunk_size, chunk_overlap).retrieve(question, k, mode)


if __name__ == "__main__":
    for q in [
        "What is faithfulness in RAG evaluation?",
        "How does an LLM judge score answers?",
        "What is the effect of chunk size on retrieval?",
    ]:
        print(f"\nQ: {q}")
        for hit in retrieve(q, k=3):
            snippet = hit["text"][:80].replace("\n", " ")
            print(f" {hit['rank']}. {hit['score']:.3f} {hit['doc_id']} p{hit['page']} | {snippet}")
