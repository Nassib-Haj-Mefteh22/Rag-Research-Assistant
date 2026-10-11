import json

from rag.evaluate import abstention_report, is_abstention
from rag.generate import answer

K = 4

if __name__ == "__main__":
    qa = [json.loads(line) for line in open("eval/qa.jsonl", encoding="utf-8")]
    rows = []
    for item in qa:
        text = answer(item["question"], k=K)["answer"]
        rows.append({"id": item["id"], "type": item["type"], "abstained": is_abstention(text)})
    rep = abstention_report(rows)
    print(f"answerable: {rep['n_answerable']} | unanswerable: {rep['n_unanswerable']}")
    print(f"correct abstention (unanswerable refused): {rep['correct_abstention']:.3f}")
    print(f"over-refusal (answerable refused):         {rep['over_refusal']:.3f}")
    print(f"accuracy: {rep['accuracy']:.3f}")
    wrong = [r["id"] for r in rows if r["type"] != "unanswerable" and r["abstained"]]
    missed = [r["id"] for r in rows if r["type"] == "unanswerable" and not r["abstained"]]
    print("over-refused:", wrong)
    print("unanswerable but answered:", missed)
