from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import load_settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(load_settings().embedding_model)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )


def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    return embed_texts(model, [QUERY_PREFIX + question])[0]


def embeddings_path(chunk_size: int, chunk_overlap: int) -> Path:
    return load_settings().processed_dir / f"embeddings_{chunk_size}_{chunk_overlap}.npy"


def embed_chunks(
    model: SentenceTransformer, chunks: list[dict], path: Path, force: bool = False
) -> np.ndarray:
    if path.exists() and not force:
        vectors = np.load(path)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"{path.name} has {len(vectors)} rows but there are {len(chunks)} chunks: "
                "stale cache, rerun with force=True"
            )
        return vectors
    vectors = embed_texts(model, [c["text"] for c in chunks])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


if __name__ == "__main__":
    import time

    from rag.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, chunks_path, load_pages

    chunks = load_pages(chunks_path(DEFAULT_SIZE, DEFAULT_OVERLAP))
    out = embeddings_path(DEFAULT_SIZE, DEFAULT_OVERLAP)
    embedder = load_embedder()

    start = time.perf_counter()
    vectors = embed_chunks(embedder, chunks, out)
    print(f"1st call: {time.perf_counter() - start:.1f}s | shape {vectors.shape} | file {out.name}")
    start = time.perf_counter()
    again = embed_chunks(embedder, chunks, out)
    print(f"2nd call (from cache): {time.perf_counter() - start:.2f}s | identical: "
          f"{np.array_equal(vectors, again)}")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"rows == chunks: {len(vectors) == len(chunks)} | norm min {norms.min():.4f} "
          f"max {norms.max():.4f}")

    twice = embed_texts(embedder, [chunks[0]["text"], chunks[0]["text"]])
    print(f"same text twice, max difference: {np.abs(twice[0] - twice[1]).max():.2e}")
    query = embed_query(embedder, "What is retrieval augmented generation?")
    for i in np.argsort(-(vectors @ query))[:3]:
        print(f"  {vectors[i] @ query:.3f} {chunks[i]['chunk_id']}: {chunks[i]['text'][:70]!r}")
