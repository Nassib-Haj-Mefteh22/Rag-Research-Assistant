import json
import sys
from pathlib import Path

from rag.retrieve import retrieve

QA_PATH = Path("eval/qa.jsonl")
TYPES = {"factual", "multi", "unanswerable"}
FIELDS = ("id", "type", "subtype", "question", "answer", "gold")


def check_row(row: dict) -> list[str]:
    problems = [f"missing field {f}" for f in FIELDS if f not in row]
    if problems:
        return problems
    if row["type"] not in TYPES:
        problems.append(f"unknown type {row['type']}")
    passages = 0
    for g in row["gold"]:
        pages = g.get("pages")
        if not g.get("doc_id") or not isinstance(pages, list) or not pages:
            problems.append(f"bad gold entry {g}")
        else:
            passages += len(pages)
    if row["type"] == "multi" and passages < 2:
        problems.append("multi question needs 2+ gold (doc, page) passages")
    if row["type"] == "factual" and passages < 1:
        problems.append("factual question needs gold")
    if row["type"] == "unanswerable" and row["gold"]:
        problems.append("unanswerable question must have empty gold")
    return problems


def show_retrieval(row: dict, k: int) -> None:
    hits = retrieve(row["question"], k=k)
    found_pages = {(h["doc_id"], h["page"]) for h in hits}
    print(f"\n=== {row['id']} [{row['subtype']}] {row['question']}")
    for h in hits:
        print(f" {h['rank']}. {h['score']:.3f} {h['doc_id']} p{h['page']}")
    gold_pairs = [(g["doc_id"], p) for g in row["gold"] for p in g["pages"]]
    found = sum(pair in found_pages for pair in gold_pairs)
    for doc_id, page in gold_pairs:
        print(f"   gold {doc_id} p{page}: {'FOUND' if (doc_id, page) in found_pages else 'missed'}")
    print(f"   -> {found}/{len(gold_pairs)} gold passages in top-{k}")


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [json.loads(line) for line in QA_PATH.read_text(encoding="utf-8").splitlines() if line]
    ids = [r.get("id") for r in rows]
    print(f"{len(rows)} rows; types: { {t: sum(r.get('type') == t for r in rows) for t in TYPES} }")
    if len(set(ids)) != len(ids):
        print("PROBLEM: duplicate ids")
    bad = 0
    for row in rows:
        for problem in check_row(row):
            bad += 1
            print(f"PROBLEM {row.get('id')}: {problem}")
    print(f"schema problems: {bad}")
    for row in rows:
        if row.get("type") == "multi" and not check_row(row):
            show_retrieval(row, k)


if __name__ == "__main__":
    main()
