from rag.generate import answer

QUESTIONS = [
    ("off-topic", "Who won the 2022 football World Cup?"),
    ("on-topic, absent", "What accuracy did GPT-7 reach on the Bamboo-RAG benchmark?"),
]

if __name__ == "__main__":
    for kind, q in QUESTIONS:
        r = answer(q)
        print(f"\n[{kind}] Q: {q}")
        print(f"A: {r['answer']}")
        top1 = r["chunks"][0]["score"]
        print(f"is_refusal={r['is_refusal']} | sources={r['sources']} | top1 score={top1:.3f}")
