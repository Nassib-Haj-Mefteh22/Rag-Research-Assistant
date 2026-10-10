"""Step 7: retrieve(question, k) -> list of chunks with scores and metadata.

WHY: steps 8-11 should call ONE function instead of repeating embed -> reshape -> search -> map.
Later variants (BM25, reranker) change the inside of this module, not its callers.
"""

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
HYBRID_POOL = 20  # candidates taken from each method before fusion


class Retriever:
    """Holds the loaded index, chunks and query embedder so they are loaded only once.

    WHY a class: loading the model and index takes seconds; calling retrieve() many times
    (10 questions now, ~40 in the eval) must not repeat that. Dependencies are passed in,
    so tests can use a tiny fake index and embedder without downloading bge.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict],
        embed: Callable[[str], np.ndarray],
        reranker: Reranker | None = None,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embed = embed  # question -> (384,) normalized vector
        self._bm25: Bm25Index | None = None
        self.reranker = reranker or Reranker()  # cheap: the model loads on first score()

    @classmethod
    def from_config(cls, chunk_size: int = DEFAULT_SIZE, chunk_overlap: int = DEFAULT_OVERLAP):
        """Build a Retriever from the files made in steps 4-6 for one chunking config."""
        chunks = load_pages(chunks_path(chunk_size, chunk_overlap))
        index = load_index(index_path(chunk_size, chunk_overlap), chunks)  # stale check inside
        model = load_embedder()
        return cls(index, chunks, lambda question: embed_query(model, question))

    def retrieve(self, question: str, k: int = DEFAULT_K, mode: str = "dense") -> list[dict]:
        """Top-k chunks for a question, best first.

        Each dict: rank (1-based), score, doc_id, page, chunk_id, text.
        mode "dense": score is the cosine. "hybrid" (7.3): dense + BM25 fused with RRF, score is
        the RRF score. Adding "+rerank" (7.4) re-sorts a pool of 20 with the cross-encoder, score
        is then its logit. Scores of different modes are not comparable.
        We copy the chunk dict so callers cannot mutate the shared chunks list.
        """
        if not question.strip():
            raise ValueError("question is empty")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        base, _, rerank = mode.partition("+")
        if base not in ("dense", "hybrid") or rerank not in ("", "rerank"):
            raise ValueError(
                f"mode must be dense, hybrid, dense+rerank or hybrid+rerank, got {mode!r}"
            )
        # With reranking we first fetch a pool of candidates, then keep the best k after re-sorting.
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
        """Re-sort (score, chunk) candidates by cross-encoder score, best first."""
        scores = self.reranker.score(question, [chunk["text"] for _, chunk in hits])
        ranked = sorted(zip(scores, (chunk for _, chunk in hits), strict=True),
                        key=lambda pair: pair[0], reverse=True)
        return [(score, chunk) for score, chunk in ranked]

    def _hybrid(self, question: str, k: int) -> list[tuple[float, dict]]:
        """Fuse the top HYBRID_POOL of dense search and of BM25, keep the k best by RRF.

        WHY a pool larger than k: a chunk ranked 12th by one method and 3rd by the other
        should still be able to win, so each method contributes more candidates than we keep.
        """
        if self._bm25 is None:  # built on first use only: tokenizing ~5k chunks takes seconds
            self._bm25 = Bm25Index(self.chunks)
        dense = search(self.index, self.chunks, self.embed(question), HYBRID_POOL)
        dense_ids = [chunk["chunk_id"] for _, chunk in dense]
        bm25_ids = [self.chunks[i]["chunk_id"] for i in self._bm25.top(question, HYBRID_POOL)]
        by_id = {c["chunk_id"]: c for c in self.chunks}
        fused = rrf([dense_ids, bm25_ids])[:k]
        return [(score, by_id[chunk_id]) for chunk_id, score in fused]


@cache
def _default_retriever(chunk_size: int, chunk_overlap: int) -> Retriever:
    """One cached Retriever per chunking config (experiments in step 11 vary the config)."""
    return Retriever.from_config(chunk_size, chunk_overlap)


def retrieve(
    question: str,
    k: int = DEFAULT_K,
    chunk_size: int = DEFAULT_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP,
    mode: str = "dense",
) -> list[dict]:
    """Entry point for steps 8+: the first call loads everything, later calls reuse it."""
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
