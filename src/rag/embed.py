"""Step 5: turn text into normalized 384-d vectors with bge-small.

WHY: search compares vectors, not strings. Chunks and questions must be embedded by the same
model, with the same normalization, or their similarities are meaningless.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import load_settings

# bge is asymmetric: only the QUERY gets this instruction (see the model card), never the chunks.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32  # only affects speed and memory, not the resulting vectors


def load_embedder() -> SentenceTransformer:
    """Load the model named in config once; reloading it per call would waste seconds each time."""
    return SentenceTransformer(load_settings().embedding_model)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed documents (chunks): no prefix. Returns (n, 384), each row of length 1.

    WHY normalize: with length-1 vectors the inner product equals cosine similarity, which is
    what FAISS IndexFlatIP will compute in step 6.
    """
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )


def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    """Embed one question with the bge query prefix. Returns (384,)."""
    return embed_texts(model, [QUERY_PREFIX + question])[0]


def embeddings_path(chunk_size: int, chunk_overlap: int) -> Path:
    """One .npy per chunking config, named like the chunks file it is aligned with."""
    return load_settings().processed_dir / f"embeddings_{chunk_size}_{chunk_overlap}.npy"


def embed_chunks(
    model: SentenceTransformer, chunks: list[dict], path: Path, force: bool = False
) -> np.ndarray:
    """Embed all chunks once and cache them; row i of the result is chunks[i].

    WHY cache: embedding ~5000 chunks takes minutes and the result never changes for a given
    chunks file, so steps 6-11 reload it in under a second instead.
    WHY the row-count check: if the chunks file changed but the .npy did not, row i would
    belong to a different chunk and citations would silently point at the wrong text.
    """
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

    chunks = load_pages(chunks_path(DEFAULT_SIZE, DEFAULT_OVERLAP))  # same jsonl format
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

    # Same text twice must give the same vector (the model is deterministic at inference).
    twice = embed_texts(embedder, [chunks[0]["text"], chunks[0]["text"]])
    print(f"same text twice, max difference: {np.abs(twice[0] - twice[1]).max():.2e}")
    query = embed_query(embedder, "What is retrieval augmented generation?")
    for i in np.argsort(-(vectors @ query))[:3]:
        print(f"  {vectors[i] @ query:.3f} {chunks[i]['chunk_id']}: {chunks[i]['text'][:70]!r}")
