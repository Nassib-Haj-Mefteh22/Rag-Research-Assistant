import pytest

from rag.hybrid import Bm25Index, rrf, tokenize


def test_tokenize_keeps_numbers_and_lowercases():
    assert tokenize("141 annotated RAGAs, recall@k") == ["141", "annotated", "ragas", "recall", "k"]


def test_rrf_rewards_agreement_between_lists():
    fused = dict(rrf([["a", "b", "c"], ["c", "b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 62)
    assert fused["a"] == pytest.approx(1 / 61)
    ordered = [item for item, _ in rrf([["a", "b", "c"], ["c", "b", "d"]])]
    assert ordered[0] == "b" or ordered[0] == "c"  # appear in both lists, so they beat a and d
    assert set(ordered[:2]) == {"b", "c"}


def test_bm25_finds_exact_token_and_drops_zero_scores():
    chunks = [{"text": "dense vectors capture meaning"}, {"text": "we annotated 141 responses"},
              {"text": "unrelated words here"}]
    index = Bm25Index(chunks)
    assert index.top("how many 141", 3) == [1]  # only chunk 1 shares a token
    assert index.top("zzz", 3) == []
