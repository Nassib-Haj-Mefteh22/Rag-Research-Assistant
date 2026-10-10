"""10.4: faithfulness judge on 10 answers, printed for a manual spot check.

For each answer: the answer text, the passages it cites (so you can verify the claims by eye)
and the judge's verdict. Refusals are skipped by the judge (None). Cached, reruns are free.
All 8 unanswerable questions are refused at k=4, so none of them appear here.
"""

import json
import re

from rag.evaluate import judge_faithfulness
from rag.generate import answer

K = 4
IDS = ["f02", "f04", "f06", "f08", "f10", "f12", "m02", "m04", "m06", "m08",
       "f14", "f15", "f17", "f18"]  # fmt: skip
# Reviewer scores after reading the printout. Claude's independent read, NOT a human's.
MY_SCORES: dict[str, int] = {
    "f02": 2, "f06": 2, "f08": 2, "f12": 2, "m02": 1, "m04": 1,
    "f14": 2, "f15": 2, "f17": 2, "f18": 2,
}  # fmt: skip

if __name__ == "__main__":
    rows = {r["id"]: r for r in map(json.loads, open("eval/qa.jsonl", encoding="utf-8"))}
    judged: dict[str, int] = {}
    for qid in IDS:
        row = rows[qid]
        result = answer(row["question"], k=K)
        verdict = judge_faithfulness(result["answer"], result["chunks"])
        if verdict is None:
            print(f"\n== {qid} ({row['type']}) REFUSAL, not judged")
            continue
        judged[qid] = verdict["score"]
        print(f"\n== {qid} ({row['type']}) judge={verdict['score']}")
        print(f"Q:   {row['question']}")
        print(f"RAG: {result['answer']}")
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])})
        for n in cited:
            if 1 <= n <= len(result["chunks"]):
                c = result["chunks"][n - 1]
                print(f"  [{n}] ({c['doc_id']}, p.{c['page']}) {c['text'][:600]!r}")
        print(f"WHY: {verdict['reason']}")
    if MY_SCORES:
        both = [q for q in judged if q in MY_SCORES]
        exact = sum(judged[q] == MY_SCORES[q] for q in both)
        near = sum(abs(judged[q] - MY_SCORES[q]) <= 1 for q in both)
        print(f"\nagreement: exact {exact}/{len(both)}, within 1: {near}/{len(both)}")
    print("\njudge distribution:", {s: list(judged.values()).count(s) for s in (0, 1, 2)})
