import argparse
import csv
import json
import re
from pathlib import Path

from rag.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE
from rag.config import ROOT, load_settings
from rag.generate import (
    REFUSAL,
    answer_from_chunks,
    answer_no_retrieval,
    format_context,
    strip_thinking,
)
from rag.llm import Messages, chat
from rag.retrieve import DEFAULT_K, retrieve


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["doc_id"] == g["doc_id"] and chunk["page"] in g["pages"] for g in gold)


def gold_passages(gold: list[dict]) -> set[tuple[str, int]]:
    return {(g["doc_id"], page) for g in gold for page in g["pages"]}


def _require_gold(gold: list[dict]) -> None:
    if not gold:
        raise ValueError("retrieval metrics are undefined for empty gold (unanswerable question)")


def hit_at_k(chunks: list[dict], gold: list[dict], k: int) -> float:
    _require_gold(gold)
    return float(any(is_hit(c, gold) for c in chunks[:k]))


def recall_at_k(chunks: list[dict], gold: list[dict], k: int) -> float:
    _require_gold(gold)
    needed = gold_passages(gold)
    found = {(c["doc_id"], c["page"]) for c in chunks[:k]} & needed
    return len(found) / len(needed)


def reciprocal_rank(chunks: list[dict], gold: list[dict], k: int) -> float:
    _require_gold(gold)
    for position, chunk in enumerate(chunks[:k], start=1):
        if is_hit(chunk, gold):
            return 1.0 / position
    return 0.0


JUDGE_RUBRIC = """You grade an answer to a question about research papers by comparing it
with a reference answer written by an expert. Judge only whether the answer says the same
thing as the reference. Do not use outside knowledge and do not reward style or length.

Scores:
2 = correct: every key fact of the reference is present (numbers must match) and nothing
    contradicts it. Different wording or extra correct detail is fine.
1 = partial: some key facts are present but one is missing or imprecise, or only one of
    several required items is given.
0 = wrong: it contradicts the reference, answers a different question, or lacks the key
    fact. A reply such as "I don't know" is 0.

Reply with exactly two lines:
SCORE: <0, 1 or 2>
REASON: <one sentence>"""


def build_judge_messages(question: str, gold_answer: str, answer_text: str) -> Messages:
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {gold_answer}\n\n"
        f"Answer to grade: {answer_text}"
    )
    return [{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": user}]


def parse_score(text: str) -> int:
    match = re.search(r"SCORE:\s*([012])\b", text)
    if match is None:
        raise ValueError(f"no valid 'SCORE: 0|1|2' in judge reply: {text[:100]!r}")
    return int(match.group(1))


def judge_correctness(question: str, gold_answer: str, answer_text: str) -> dict:
    messages = build_judge_messages(question, gold_answer, answer_text)
    reply = strip_thinking(chat(load_settings().judge_model, messages, temperature=0.0))
    reason = re.search(r"REASON:\s*(.+)", reply, flags=re.DOTALL)
    return {"score": parse_score(reply), "reason": reason.group(1).strip() if reason else ""}


FAITHFULNESS_RUBRIC = """You check whether an answer is supported by the numbered context
passages it was written from. Use ONLY the passages: ignore outside knowledge, and do not
judge whether the answer is correct or complete. A claim is supported if any passage states it.
A sentence saying the passages do not contain or specify something is not a claim: ignore it.

Scores:
2 = faithful: every claim and number in the answer is stated in the passages.
1 = partly faithful: most claims are supported, but at least one claim or number is not
    in the passages.
0 = unfaithful: the main claim is missing from the passages or contradicts them.

Reply with exactly two lines:
SCORE: <0, 1 or 2>
REASON: <one sentence naming the unsupported claim, if any>"""


def build_faithfulness_messages(answer_text: str, chunks: list[dict]) -> Messages:
    user = f"Passages:\n{format_context(chunks)}\n\nAnswer to check: {answer_text}"
    return [
        {"role": "system", "content": FAITHFULNESS_RUBRIC},
        {"role": "user", "content": user},
    ]


def judge_faithfulness(answer_text: str, chunks: list[dict]) -> dict | None:
    if is_abstention(answer_text):
        return None
    messages = build_faithfulness_messages(answer_text, chunks)
    reply = strip_thinking(chat(load_settings().judge_model, messages, temperature=0.0))
    reason = re.search(r"REASON:\s*(.+)", reply, flags=re.DOTALL)
    return {"score": parse_score(reply), "reason": reason.group(1).strip() if reason else ""}


def is_abstention(answer_text: str) -> bool:
    return answer_text.strip() == REFUSAL


def abstention_report(rows: list[dict]) -> dict:
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answerable = [r for r in rows if r["type"] != "unanswerable"]

    def rate(group: list[dict]) -> float | None:
        return sum(r["abstained"] for r in group) / len(group) if group else None

    refused_ok = sum(r["abstained"] for r in unanswerable)
    answered_ok = sum(not r["abstained"] for r in answerable)
    correct = refused_ok + answered_ok
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "correct_abstention": rate(unanswerable),
        "over_refusal": rate(answerable),
        "accuracy": correct / len(rows) if rows else None,
    }


def _judge_or_error(call, *args) -> tuple[dict | None, str | None]:
    try:
        return call(*args), None
    except ValueError as exc:
        return None, str(exc)


def evaluate_item(item: dict, k: int, chunk_size: int, chunk_overlap: int, mode: str) -> dict:
    no_retrieval = mode == "none"
    if no_retrieval:
        chunks: list[dict] = []
        result = answer_no_retrieval(item["question"])
    else:
        chunks = retrieve(item["question"], k, chunk_size, chunk_overlap, mode)
        result = answer_from_chunks(item["question"], chunks)
    text = result["answer"]
    abstained = is_abstention(text)
    answerable = item["type"] != "unanswerable"
    row: dict = {
        "id": item["id"],
        "type": item["type"],
        "answer": text,
        "abstained": abstained,
        "retrieved": [(c["doc_id"], c["page"]) for c in chunks],
        "hit": None,
        "recall": None,
        "rr": None,
        "correctness": None,
        "correctness_reason": None,
        "faithfulness": None,
        "faithfulness_reason": None,
        "judge_errors": [],
    }
    if answerable:
        gold = item["gold"]
        if not no_retrieval:
            row["hit"] = hit_at_k(chunks, gold, k)
            row["recall"] = recall_at_k(chunks, gold, k)
            row["rr"] = reciprocal_rank(chunks, gold, k)
        if abstained:
            row["correctness"], row["correctness_reason"] = 0, "refusal on an answerable question"
        else:
            verdict, error = _judge_or_error(
                judge_correctness, item["question"], item["answer"], text
            )
            if verdict:
                row["correctness"], row["correctness_reason"] = verdict["score"], verdict["reason"]
            else:
                row["judge_errors"].append(error)
    if not abstained and chunks:
        verdict, error = _judge_or_error(judge_faithfulness, text, chunks)
        if verdict:
            row["faithfulness"], row["faithfulness_reason"] = verdict["score"], verdict["reason"]
        else:
            row["judge_errors"].append(error)
    return row


def _mean(values: list[float]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    full = [r["correctness"] == 2 for r in answerable if r["correctness"] is not None]
    return {
        "hit": _mean([r["hit"] for r in rows]),
        "recall": _mean([r["recall"] for r in rows]),
        "mrr": _mean([r["rr"] for r in rows]),
        "correct_mean": _mean([r["correctness"] for r in answerable]),
        "correct_full": sum(full) / len(full) if full else None,
        "faith_mean": _mean([r["faithfulness"] for r in rows]),
        "faith_n": sum(r["faithfulness"] is not None for r in rows),
        **{
            key: value
            for key, value in abstention_report(rows).items()
            if key not in ("n_answerable", "n_unanswerable")
        },
        "n_answerable": len(answerable),
        "n_unanswerable": len(rows) - len(answerable),
        "judge_errors": sum(len(r["judge_errors"]) for r in rows),
    }


def write_results(name: str, config: dict, rows: list[dict], results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"name": name, **config, **summarize(rows)}
    path = results_dir / "summary.csv"
    existing: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["name"] != name]
    existing.append({key: _fmt(value) for key, value in summary.items()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerows(existing)
    return summary


def _fmt(value: object) -> object:
    if value is None:
        return ""
    return round(value, 3) if isinstance(value, float) else value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on eval/qa.jsonl")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", default="dense")
    parser.add_argument("--name", default=None, help="default: cs<size>_ov<overlap>_k<k>_<mode>")
    args = parser.parse_args(argv)
    name = args.name or f"cs{args.chunk_size}_ov{args.overlap}_k{args.k}_{args.mode}"
    qa_path = ROOT / "eval" / "qa.jsonl"
    items = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for i, item in enumerate(items, 1):
        rows.append(evaluate_item(item, args.k, args.chunk_size, args.overlap, args.mode))
        print(f"[{i}/{len(items)}] {item['id']}", end="\r", flush=True)
    config = {
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "k": args.k,
        "mode": args.mode,
    }
    summary = write_results(name, config, rows, ROOT / "results")
    print(f"\nconfig {name}: {len(rows)} questions -> results/{name}.jsonl, results/summary.csv")
    for key, value in summary.items():
        if key != "name":
            print(f"  {key:>20}: {_fmt(value)}")


if __name__ == "__main__":
    main()
