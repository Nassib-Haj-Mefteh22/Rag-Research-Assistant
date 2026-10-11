import re

from rag.config import load_settings
from rag.llm import Messages, chat
from rag.retrieve import DEFAULT_K, retrieve

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions about research papers using ONLY the numbered
context passages.
Rules:
1. Use only facts stated in the context. Do not use outside knowledge.
2. After each claim, cite the passage number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply exactly: {REFUSAL}
4. Be concise: at most 4 sentences."""


NO_RAG_PROMPT = """You answer questions about recent research papers from your own knowledge.
No documents are provided. Be concise: at most 4 sentences. If you do not know, say so."""


def format_context(chunks: list[dict]) -> str:
    blocks = [
        f"[{n}] ({chunk['doc_id']}, p.{chunk['page']})\n{chunk['text']}"
        for n, chunk in enumerate(chunks, 1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> Messages:
    user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_citations(answer: str, n_chunks: int) -> tuple[list[int], list[int]]:
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    valid = sorted(n for n in found if 1 <= n <= n_chunks)
    invalid = sorted(n for n in found if not 1 <= n <= n_chunks)
    return valid, invalid


def answer_from_chunks(question: str, chunks: list[dict]) -> dict:
    text = chat(load_settings().gen_model, build_messages(question, chunks), temperature=0.0)
    text = strip_thinking(text)
    valid, invalid = extract_citations(text, len(chunks))
    return {
        "question": question,
        "answer": text,
        "chunks": chunks,
        "sources": [(chunks[n - 1]["doc_id"], chunks[n - 1]["page"]) for n in valid],
        "invalid_citations": invalid,
        "is_refusal": REFUSAL in text,
    }


def answer_no_retrieval(question: str) -> dict:
    messages: Messages = [
        {"role": "system", "content": NO_RAG_PROMPT},
        {"role": "user", "content": question},
    ]
    text = strip_thinking(chat(load_settings().gen_model, messages, temperature=0.0))
    return {
        "question": question,
        "answer": text,
        "chunks": [],
        "sources": [],
        "invalid_citations": [],
        "is_refusal": REFUSAL in text,
    }


def answer(question: str, k: int = DEFAULT_K, mode: str = "dense") -> dict:
    return answer_from_chunks(question, retrieve(question, k=k, mode=mode))


if __name__ == "__main__":
    for q in [
        "What is faithfulness in RAG evaluation?",
        "How does an LLM judge score answers?",
        "What is the effect of chunk size on retrieval?",
    ]:
        result = answer(q)
        print(f"\nQ: {q}\nA: {result['answer']}")
        print(f"sources: {result['sources']} | invalid: {result['invalid_citations']}")
