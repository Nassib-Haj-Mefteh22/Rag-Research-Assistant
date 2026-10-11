import json

from rag.evaluate import judge_correctness
from rag.generate import answer

K = 4
IDS = ["f01", "f03", "f05", "f07", "f09", "f11", "f13", "f16", "f19", "f22",
       "m01", "m03", "m05", "m07", "m09"]
MY_SCORES: dict[str, int] = {
    "f01": 2, "f03": 2, "f05": 2, "f07": 2, "f09": 2, "f11": 2, "f13": 2, "f16": 2,
    "f19": 1, "f22": 2, "m01": 0, "m03": 1, "m05": 1, "m07": 1, "m09": 0,
}

if __name__ == "__main__":
    rows = {r["id"]: r for r in map(json.loads, open("eval/qa.jsonl", encoding="utf-8"))}
    judged = {}
    for qid in IDS:
        row = rows[qid]
        result = answer(row["question"], k=K)
        verdict = judge_correctness(row["question"], row["answer"], result["answer"])
        judged[qid] = verdict["score"]
        print(f"\n== {qid} ({row['type']}) judge={verdict['score']}")
        print(f"Q:    {row['question']}")
        print(f"GOLD: {row['answer']}")
        print(f"RAG:  {result['answer']}")
        print(f"WHY:  {verdict['reason']}")
    if MY_SCORES:
        both = [q for q in IDS if q in MY_SCORES]
        exact = sum(judged[q] == MY_SCORES[q] for q in both)
        near = sum(abs(judged[q] - MY_SCORES[q]) <= 1 for q in both)
        print(f"\nagreement: exact {exact}/{len(both)}, within 1: {near}/{len(both)}")
    print("\njudge distribution:", {s: list(judged.values()).count(s) for s in (0, 1, 2)})
