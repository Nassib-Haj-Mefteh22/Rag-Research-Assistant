import numpy as np
import pytest

from rag.index import DIM, build_index, load_index, save_index, search


def unit(*values: float) -> np.ndarray:
    v = np.zeros(DIM)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


CHUNKS = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
VECTORS = np.stack([unit(1, 0), unit(1, 1), unit(0, 1)])


def test_search_is_best_first_and_score_is_dot_product():
    index = build_index(VECTORS)
    query = unit(1, 0)
    results = search(index, CHUNKS, query, k=3)
    assert [c["text"] for _, c in results] == ["a", "b", "c"]
    scores = [s for s, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[1] == pytest.approx(float(VECTORS[1] @ query), abs=1e-6)


def test_save_load_gives_same_results(tmp_path):
    index = build_index(VECTORS)
    path = tmp_path / "x.index"
    save_index(index, path)
    reloaded = load_index(path, CHUNKS)
    assert search(reloaded, CHUNKS, unit(0, 1), 2) == search(index, CHUNKS, unit(0, 1), 2)


def test_load_rejects_stale_index(tmp_path):
    path = tmp_path / "x.index"
    save_index(build_index(VECTORS), path)
    with pytest.raises(ValueError, match="stale"):
        load_index(path, CHUNKS[:2])


def test_build_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        build_index(np.zeros((3, 10)))


def test_numpy_search_matches_faiss_on_toy_data():
    from rag.index import numpy_search

    query = unit(1, 0.2)
    f_scores, f_ids = build_index(VECTORS).search(query.astype(np.float32).reshape(1, -1), 3)
    n_scores, n_ids = numpy_search(VECTORS, query, 3)
    assert list(f_ids[0]) == list(n_ids)
    assert np.allclose(f_scores[0], n_scores, atol=1e-6)
