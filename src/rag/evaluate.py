"""Step 10: retrieval metrics, LLM-judge metrics, results table."""

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
    """True if the chunk lies on a gold (doc_id, page).

    WHY gold is (doc_id, pages) and not chunk ids: chunk ids change with every chunk_size
    in step 11, but the paper and page holding the answer never do. Chunks are split per
    page (see chunk.py), so a chunk has exactly one page and strict matching is well defined.
    Empty gold (unanswerable question) never matches.
    """
    return any(chunk["doc_id"] == g["doc_id"] and chunk["page"] in g["pages"] for g in gold)


def gold_passages(gold: list[dict]) -> set[tuple[str, int]]:
    """Flatten gold into the set of (doc_id, page) passages that must be found.

    WHY: Recall@k counts passages, and m01-style questions list several pages per doc.
    """
    return {(g["doc_id"], page) for g in gold for page in g["pages"]}


def _require_gold(gold: list[dict]) -> None:
    """Unanswerable questions have no passage to find; the caller must skip them."""
    if not gold:
        raise ValueError("retrieval metrics are undefined for empty gold (unanswerable question)")


def hit_at_k(chunks: list[dict], gold: list[dict], k: int) -> float:
    """1.0 if any of the top-k chunks is a hit, else 0.0: 'did we find it at all?'"""
    _require_gold(gold)
    return float(any(is_hit(c, gold) for c in chunks[:k]))


def recall_at_k(chunks: list[dict], gold: list[dict], k: int) -> float:
    """Share of gold passages (doc, page) covered by the top-k: 'did we find all of it?'

    Equals hit_at_k for one-passage questions; differs on multi-passage ones
    (2 pages, 1 found = 0.5).
    """
    _require_gold(gold)
    needed = gold_passages(gold)
    found = {(c["doc_id"], c["page"]) for c in chunks[:k]} & needed
    return len(found) / len(needed)


def reciprocal_rank(chunks: list[dict], gold: list[dict], k: int) -> float:
    """1 / rank of the first hit in the top-k, 0.0 if none: 'how high did it rank?'

    Averaged over questions this is MRR. Rank is the list position + 1, not chunk['rank'],
    so the metric does not depend on a field a retriever might forget to set.
    """
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
    """Judge prompt: rubric as system message, the three texts labelled in the user message.

    WHY no retrieved chunks: correctness compares answer vs gold only; whether the answer is
    grounded in the context is faithfulness (10.4), a separate question.
    """
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {gold_answer}\n\n"
        f"Answer to grade: {answer_text}"
    )
    return [{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": user}]


def parse_score(text: str) -> int:
    """Extract the 0/1/2 after 'SCORE:'; raise on anything else.

    WHY strict: a malformed judge reply must never be silently scored (as 0 it would look
    like a wrong answer and bias every experiment).
    """
    match = re.search(r"SCORE:\s*([012])\b", text)
    if match is None:
        raise ValueError(f"no valid 'SCORE: 0|1|2' in judge reply: {text[:100]!r}")
    return int(match.group(1))


def judge_correctness(question: str, gold_answer: str, answer_text: str) -> dict:
    """Ask the judge model (different from the generator, temperature 0, cached).

    Returns {"score": 0|1|2, "reason": str}.
    """
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
    """Judge prompt: rubric as system message, then the passages exactly as the generator saw them.

    WHY format_context: same [n] (doc, page) layout as generation, so the judge reads what
    the generator read and nothing else (no gold answer: faithfulness is gold-free).
    """
    user = f"Passages:\n{format_context(chunks)}\n\nAnswer to check: {answer_text}"
    return [
        {"role": "system", "content": FAITHFULNESS_RUBRIC},
        {"role": "user", "content": user},
    ]


def judge_faithfulness(answer_text: str, chunks: list[dict]) -> dict | None:
    """Faithfulness 0/1/2 of an answer w.r.t. its chunks, or None for a refusal.

    WHY None: a refusal makes no claim, so it is neither faithful nor unfaithful; counting it
    as 2 or 0 would distort the average. Whether refusing was right is scored in 10.5.
    """
    if is_abstention(answer_text):
        return None
    messages = build_faithfulness_messages(answer_text, chunks)
    reply = strip_thinking(chat(load_settings().judge_model, messages, temperature=0.0))
    reason = re.search(r"REASON:\s*(.+)", reply, flags=re.DOTALL)
    return {"score": parse_score(reply), "reason": reason.group(1).strip() if reason else ""}


def is_abstention(answer_text: str) -> bool:
    """True if the whole answer is the fixed refusal sentence of generate.py.

    WHY string match: the prompt forces this exact sentence, so no judge is needed. A hedge
    like 'the documents do not specify X' inside a real answer is NOT an abstention.
    WHY equality, not 'contains' (fixed in 11.7): a real answer with the sentence appended
    (step 11.6, m09) used to count as a refusal, scored 0 and skipped by the faithfulness judge.
    """
    return answer_text.strip() == REFUSAL


def abstention_report(rows: list[dict]) -> dict:
    """Abstention scores from rows {'type': 'factual'|'multi'|'unanswerable', 'abstained': bool}.

    Two rates, because one number hides a trade-off (always refusing scores 100% on the
    unanswerable ones): correct_abstention should be high, over_refusal should be low.
    A rate is None when its group is empty.
    """
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
    """Run a judge; a malformed reply becomes (None, message) instead of aborting the run.

    WHY: the reply is cached on disk, so a crash would repeat forever. The error is stored in
    the row and counted in the summary, so it is visible and never silently scored.
    """
    try:
        return call(*args), None
    except ValueError as exc:
        return None, str(exc)


def evaluate_item(item: dict, k: int, chunk_size: int, chunk_overlap: int, mode: str) -> dict:
    """Run one eval question through RAG and score it. Which scores apply depends on the type.

    - retrieval metrics: answerable questions only (unanswerable have no gold passage)
    - correctness: answerable and answered; a refusal on an answerable question is 0 by the
      rubric, so it is set without a judge call
    - faithfulness: every non-refusal (also unanswerable ones the model answered anyway:
      that is a hallucination check); refusals are None
    """
    no_retrieval = mode == "none"  # E0 baseline: no chunks, own-knowledge prompt
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
        if not no_retrieval:  # with no passages, hit/recall/MRR stay None (not 0: n/a)
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
    if not abstained and chunks:  # faithfulness needs context to check against
        verdict, error = _judge_or_error(judge_faithfulness, text, chunks)
        if verdict:
            row["faithfulness"], row["faithfulness_reason"] = verdict["score"], verdict["reason"]
        else:
            row["judge_errors"].append(error)
    return row


def _mean(values: list[float]) -> float | None:
    """Mean of the non-None values, None if there are none."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def summarize(rows: list[dict]) -> dict:
    """One summary line from per-question rows. Each mean uses only the rows where it applies.

    correct_mean is on the 0-2 scale; correct_full is the share of answerable questions scored 2
    (a refused question counts as 0 there, so it is comparable across configs).
    """
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
    """Write results/<name>.jsonl (one line per question) and upsert the row in summary.csv.

    WHY upsert by name: rerunning a config replaces its row, so the table never has duplicates.
    """
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
    """Round floats for the CSV; None becomes an empty cell."""
    if value is None:
        return ""
    return round(value, 3) if isinstance(value, float) else value


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m rag.evaluate --k 4 [--chunk-size 1000 --overlap 100 --mode dense --name X]"""
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
