"""Step 6.2: check that FAISS and plain numpy return the same top-5 for 20 queries."""

import numpy as np

from rag.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, chunks_path, load_pages
from rag.embed import embed_query, embeddings_path, load_embedder
from rag.index import index_path, load_index, numpy_search

K = 5
QUESTIONS = [
    "What is faithfulness in RAG evaluation?",
    "How does an LLM judge score answers?",
    "What is the effect of chunk size on retrieval?",
    "What is retrieval-augmented generation?",
    "How is retrieval quality measured?",
    "What is hallucination in language models?",
    "What does recall@k measure?",
    "How do dense retrievers differ from BM25?",
    "What is a cross-encoder reranker?",
    "How are RAG benchmarks constructed?",
    "What is answer relevance?",
    "How do LLM judges agree with human annotators?",
    "What is context precision?",
    "How does multi-hop question answering work?",
    "What are the limitations of LLM-as-a-judge?",
    "How are documents split into chunks?",
    "What is position bias in judges?",
    "How can RAG systems abstain from answering?",
    "What datasets are used to evaluate retrieval?",
    "What is reciprocal rank fusion?",
]


def main() -> None:
    chunks = load_pages(chunks_path(DEFAULT_SIZE, DEFAULT_OVERLAP))
    vectors = np.load(embeddings_path(DEFAULT_SIZE, DEFAULT_OVERLAP))
    index = load_index(index_path(DEFAULT_SIZE, DEFAULT_OVERLAP), chunks)
    embedder = load_embedder()

    same_ids = same_scores = 0
    for question in QUESTIONS:
        q = embed_query(embedder, question)
        f_scores, f_ids = index.search(q.astype(np.float32).reshape(1, -1), K)
        n_scores, n_ids = numpy_search(vectors, q, K)
        same_ids += int(np.array_equal(f_ids[0], n_ids))
        same_scores += int(np.allclose(f_scores[0], n_scores, atol=1e-5))
        if not np.array_equal(f_ids[0], n_ids):
            print(f"DIFF: {question}\n  faiss {f_ids[0]}\n  numpy {n_ids}")
    print(f"{len(QUESTIONS)} queries | identical top-{K} ids: {same_ids} | "
          f"scores within 1e-5: {same_scores}")


if __name__ == "__main__":
    main()
