import sys
import time

from rag.retrieve import retrieve

QUESTIONS = [
    ("What is faithfulness in RAG evaluation?", "obvious"),
    ("How does an LLM-as-a-judge score generated answers?", "obvious"),
    ("What is the effect of chunk size on retrieval quality?", "obvious"),
    ("Which metrics measure retrieval quality, such as recall@k or MRR?", "obvious"),
    ("What is the difference between dense and sparse retrieval?", "obvious"),
    ("What is hallucination in retrieval-augmented generation?", "obvious"),
    ("How can a RAG system learn to say that it does not know?", "obvious"),
    ("How does a cross-encoder reranker improve retrieval?", "obvious"),
    ("How many annotated responses were in the evaluation dataset?", "number"),
    ("What is RAGAS?", "acronym"),
]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dense"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for number, (question, kind) in enumerate(QUESTIONS, 1):
        print(f"\n=== Q{number} [{kind}] {question}")
        start = time.perf_counter()
        hits = retrieve(question, k=5, mode=mode)
        print(f" ({time.perf_counter() - start:.2f}s)")
        for hit in hits:
            snippet = hit["text"][:110].replace("\n", " ")
            print(f" {hit['rank']}. {hit['score']:.3f} {hit['doc_id']} p{hit['page']} | {snippet}")


if __name__ == "__main__":
    main()
