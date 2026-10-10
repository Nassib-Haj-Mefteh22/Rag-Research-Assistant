import numpy as np
import pytest

from rag.embed import QUERY_PREFIX, embed_chunks, embed_query, embed_texts


class FakeModel:
    """Stands in for SentenceTransformer so tests need no model download."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.seen = list(texts)
        vectors = np.array([[len(t), 1.0, 2.0] for t in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_query_gets_prefix_and_documents_do_not():
    model = FakeModel()
    embed_query(model, "what is RAG?")
    assert model.seen == [QUERY_PREFIX + "what is RAG?"]
    embed_texts(model, ["a chunk"])
    assert model.seen == ["a chunk"]


def test_shapes_and_unit_norm():
    vectors = embed_texts(FakeModel(), ["one", "two words", "three"])
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert embed_query(FakeModel(), "q").shape == (3,)


CHUNKS = [{"text": "alpha"}, {"text": "beta beta"}]


def test_embed_chunks_caches_and_reloads(tmp_path):
    path = tmp_path / "emb.npy"
    first = embed_chunks(FakeModel(), CHUNKS, path)
    assert path.exists() and first.shape == (2, 3)
    model = FakeModel()
    second = embed_chunks(model, CHUNKS, path)
    assert model.seen == []  # served from disk, the model was not called
    assert np.array_equal(first, second)


def test_embed_chunks_rejects_stale_cache(tmp_path):
    path = tmp_path / "emb.npy"
    embed_chunks(FakeModel(), CHUNKS, path)
    with pytest.raises(ValueError):
        embed_chunks(FakeModel(), CHUNKS + [{"text": "gamma"}], path)
