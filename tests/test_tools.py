from rag import tools

FAKE_CHUNKS = [
    {"doc_id": "2501.00001", "page": 3, "score": 0.9, "text": "alpha", "rank": 1, "chunk_id": "a"},
    {"doc_id": "9999.99999", "page": 7, "score": 0.5, "text": "beta", "rank": 2, "chunk_id": "b"},
]


def test_results_are_simple_dicts_with_titles(monkeypatch):
    seen = {}

    def fake_retrieve(query, k, mode):
        seen.update(query=query, k=k, mode=mode)
        return FAKE_CHUNKS

    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    results = tools.search_papers("What is BM25?", k=2, titles={"2501.00001": "Paper One"})
    assert seen == {"query": "What is BM25?", "k": 2, "mode": "hybrid+rerank"}
    assert set(results[0]) == {"doc_id", "title", "page", "score", "text"}
    assert results[0]["title"] == "Paper One"
    assert results[1]["title"] == "(title unknown)"  # missing metadata must not crash


def test_empty_result_gives_empty_list(monkeypatch):
    monkeypatch.setattr(tools, "retrieve", lambda query, k, mode: [])
    assert tools.search_papers("x", titles={}) == []
