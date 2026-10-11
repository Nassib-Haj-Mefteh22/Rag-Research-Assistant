import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "cs1000_ov100_k4_hybrid.jsonl"

CATEGORIES = {
    1: "Not retrieved (fact absent from top 20)",
    2: "Retrieved but ranked too low (fact in top 20, not in top 4)",
    4: "Retrieved but LLM ignored or omitted it",
    5: "Unsupported claim / judge flags faithfulness",
    7: "Wrong refusal / refusal sentence appended to a real answer",
}

LABELS: dict[str, tuple[int, str]] = {
    "f04": (2, "gold p6 absent at k=4, fact found at k=8; model refused"),
    "f18": (2, "Youden/TPR definition absent at k=4, found at k=8; model refused"),
    "f20": (2, "Retrieve.KV / En.Sum absent at k=4, found at k=8; model refused"),
    "f21": (2, "right page hit but chunk lacks the 3 types; found at k=8; model refused"),
    "f19": (2, "'91 name pairs' only found at k=20; answer partial"),
    "m02": (2, "second paper (MedCPT) absent at k=4, found at k=8; answer half"),
    "m05": (2, "stopping rule ('registry') absent at k=4, found at k=8; answer half"),
    "m06": (1, "RINSE paper never retrieved, even at k=20; answer half + refusal sentence"),
    "m07": (1, "83.7 / 77.6 not in top 20: page-level hit but wrong chunk"),
    "m08": (1, "'36%' not in top 20: page-level hit but wrong chunk"),
    "f09": (4, "'124M' is in the retrieved text; answer omitted the size"),
    "m10": (4, "79.6% is in the retrieved text; model said no aggregate share is given"),
    "m01": (5, "faithfulness 1: D1/D2 -> leave/conduct mapping not stated; details omitted"),
    "m04": (5, "faithfulness 1 but correctness 2: computed differences, judge too strict"),
    "m09": (7, "answer is right; refusal sentence appended, so it is scored as a refusal"),
}


def is_failure(row: dict) -> bool:
    if row["type"] == "unanswerable":
        return not row["abstained"]
    low_faith = row["faithfulness"] is not None and row["faithfulness"] < 2
    return (row["correctness"] or 0) < 2 or low_faith


def main() -> None:
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
    failing = [r["id"] for r in rows if is_failure(r)]
    missing = set(failing) - set(LABELS)
    extra = set(LABELS) - set(failing)
    assert not missing and not extra, f"unlabelled {sorted(missing)}, stale {sorted(extra)}"

    out = ROOT / "results" / "failures_hybrid_k4.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "category_name", "evidence"])
        for qid in failing:
            cat, evidence = LABELS[qid]
            writer.writerow([qid, cat, CATEGORIES[cat], evidence])

    counts = Counter(LABELS[q][0] for q in failing)
    print(f"{len(failing)} failures of {len(rows)} questions -> {out.name}")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>2} ({n / len(failing):.0%})  [{cat}] {CATEGORIES[cat]}")


if __name__ == "__main__":
    main()
