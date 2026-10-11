import pytest
from tests.test_index import CHUNKS, VECTORS, unit

from rag.index import build_index
from rag.retrieve import Retriever


def make_retriever() -> Retriever:
    return Retriever(build_index(VECTORS), CHUNKS, lambda question: unit(1, 0))


def test_keys_and_sorted_scores():
    hits = make_retriever().retrieve("anything", k=3)
    assert set(hits[0]) == {"text", "rank", "score"}
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert [h["text"] for h in hits] == ["a", "b", "c"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_is_respected_and_chunks_not_mutated():
    retriever = make_retriever()
    assert len(retriever.retrieve("q", k=2)) == 2
    assert "rank" not in CHUNKS[0]


def test_bad_inputs():
    retriever = make_retriever()
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ")
    with pytest.raises(ValueError, match="k must"):
        retriever.retrieve("q", k=0)


def test_hybrid_mode_returns_fused_results_and_rejects_unknown_mode():
    chunks = [{"chunk_id": "a", "text": "alpha"}, {"chunk_id": "b", "text": "beta"},
              {"chunk_id": "c", "text": "needle"}]
    retriever = Retriever(build_index(VECTORS), chunks, lambda question: unit(1, 0))
    hits = retriever.retrieve("needle", k=3, mode="hybrid")
    assert {h["chunk_id"] for h in hits} == {"a", "b", "c"}
    assert hits[0]["chunk_id"] in {"a", "c"}
    assert [h["rank"] for h in hits] == [1, 2, 3]
    with pytest.raises(ValueError, match="mode"):
        retriever.retrieve("q", mode="nope")


class FakeReranker:
    def score(self, question, texts):
        return [1.0 if t == "c" else 0.0 for t in texts]


def test_rerank_mode_reorders_candidates_and_respects_k():
    retriever = Retriever(
        build_index(VECTORS), CHUNKS, lambda question: unit(1, 0), reranker=FakeReranker()
    )
    assert retriever.retrieve("q", k=3)[0]["text"] == "a"
    hits = retriever.retrieve("q", k=2, mode="dense+rerank")
    assert [h["text"] for h in hits][0] == "c"
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
