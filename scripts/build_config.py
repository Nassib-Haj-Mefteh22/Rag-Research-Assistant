import argparse
import time

from rag.chunk import chunk_pages, chunk_stats, chunks_path, load_pages, pages_path, write_chunks
from rag.embed import embed_chunks, embeddings_path, load_embedder
from rag.index import build_index, index_path, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunks, embeddings and index for a config")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    args = parser.parse_args()
    size, overlap = args.chunk_size, args.overlap

    pages = load_pages(pages_path())
    chunks = chunk_pages(pages, size, overlap)
    write_chunks(chunks, chunks_path(size, overlap))
    stats = chunk_stats(chunks)
    print(f"pages {len(pages)} -> chunks {len(chunks)} (size {size}, overlap {overlap})")
    print(
        f"chars: mean {stats['mean']:.0f} | median {stats['median']:.0f} | "
        f"min {stats['min']} | max {stats['max']} | under 100 chars: {stats['tiny']}"
    )

    start = time.perf_counter()
    vectors = embed_chunks(load_embedder(), chunks, embeddings_path(size, overlap))
    print(f"embeddings {vectors.shape} in {time.perf_counter() - start:.0f}s")

    save_index(build_index(vectors), index_path(size, overlap))
    print(f"index written: {index_path(size, overlap).name}")


if __name__ == "__main__":
    main()
