from pathlib import Path

import faiss
import numpy as np

from rag.config import load_settings

DIM = 384


def index_path(chunk_size: int, chunk_overlap: int) -> Path:
    return load_settings().index_dir / f"faiss_{chunk_size}_{chunk_overlap}.index"


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2 or vectors.shape[1] != DIM:
        raise ValueError(f"expected shape (n, {DIM}), got {vectors.shape}")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path, chunks: list[dict]) -> faiss.Index:
    index = faiss.read_index(str(path))
    if index.ntotal != len(chunks):
        raise ValueError(
            f"{path.name} has {index.ntotal} vectors but there are {len(chunks)} chunks: "
            "stale index, rebuild it"
        )
    return index


def search(
    index: faiss.Index, chunks: list[dict], query_vec: np.ndarray, k: int
) -> list[tuple[float, dict]]:
    query = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
    scores, ids = index.search(query, k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0], strict=True) if i != -1]


def numpy_search(
    vectors: np.ndarray, query_vec: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = vectors.astype(np.float32) @ query_vec.astype(np.float32)
    top = np.argsort(-scores)[:k]
    return scores[top], top


if __name__ == "__main__":
    from rag.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, chunks_path, load_pages
    from rag.embed import embed_query, embeddings_path, load_embedder

    chunks = load_pages(chunks_path(DEFAULT_SIZE, DEFAULT_OVERLAP))
    vectors = np.load(embeddings_path(DEFAULT_SIZE, DEFAULT_OVERLAP))
    path = index_path(DEFAULT_SIZE, DEFAULT_OVERLAP)

    save_index(build_index(vectors), path)
    index = load_index(path, chunks)
    print(f"index: {index.ntotal} vectors, dim {index.d}, file {path.name}")

    embedder = load_embedder()
    for question in [
        "What is faithfulness in RAG evaluation?",
        "How does an LLM judge score answers?",
        "What is the effect of chunk size on retrieval?",
    ]:
        print(f"\nQ: {question}")
        hits = search(index, chunks, embed_query(embedder, question), 3)
        for rank, (score, chunk) in enumerate(hits, 1):
            snippet = chunk["text"][:80].replace("\n", " ")
            print(f" {rank}. {score:.3f} {chunk['doc_id']} p{chunk['page']} | {snippet}")
