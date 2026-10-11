from rag.evaluate import is_hit

GOLD = [{"doc_id": "2609.16519", "pages": [5]}]


def chunk(doc_id: str, page: int) -> dict:
    return {"doc_id": doc_id, "page": page, "text": "x"}


def test_right_doc_right_page():
    assert is_hit(chunk("2609.16519", 5), GOLD)


def test_right_doc_wrong_page():
    assert not is_hit(chunk("2609.16519", 6), GOLD)


def test_wrong_doc_right_page():
    assert not is_hit(chunk("2608.26379", 5), GOLD)


def test_multi_page_gold_any_page_counts():
    gold = [{"doc_id": "2609.31660", "pages": [3, 7]}]
    assert is_hit(chunk("2609.31660", 7), gold)
    assert not is_hit(chunk("2609.31660", 4), gold)


def test_multi_doc_gold_any_entry_counts():
    gold = GOLD + [{"doc_id": "2608.26379", "pages": [4]}]
    assert is_hit(chunk("2608.26379", 4), gold)


def test_empty_gold_never_hits():
    assert not is_hit(chunk("2609.16519", 5), [])


import pytest  # noqa: E402

from rag.evaluate import gold_passages, hit_at_k, recall_at_k, reciprocal_rank  # noqa: E402

MULTI = [{"doc_id": "D", "pages": [3, 7]}]


def test_gold_passages_flattens_pages():
    assert gold_passages(MULTI) == {("D", 3), ("D", 7)}


def test_perfect_retriever_scores_one():
    perfect = [chunk("D", 3), chunk("D", 7), chunk("X", 1)]
    assert hit_at_k(perfect, MULTI, 4) == 1.0
    assert recall_at_k(perfect, MULTI, 4) == 1.0
    assert reciprocal_rank(perfect, MULTI, 4) == 1.0


def test_useless_retriever_scores_zero():
    junk = [chunk("X", p) for p in range(1, 5)]
    assert hit_at_k(junk, MULTI, 4) == 0.0
    assert recall_at_k(junk, MULTI, 4) == 0.0
    assert reciprocal_rank(junk, MULTI, 4) == 0.0


def test_recall_partial_on_multi_passage():
    got = [chunk("X", 1), chunk("D", 7)]
    assert hit_at_k(got, MULTI, 4) == 1.0
    assert recall_at_k(got, MULTI, 4) == 0.5
    assert reciprocal_rank(got, MULTI, 4) == 0.5


def test_duplicate_page_chunks_do_not_inflate_recall():
    got = [chunk("D", 3), chunk("D", 3)]
    assert recall_at_k(got, MULTI, 4) == 0.5


def test_reciprocal_rank_is_first_hit_only():
    got = [chunk("X", 1), chunk("X", 2), chunk("D", 3), chunk("D", 7)]
    assert reciprocal_rank(got, MULTI, 4) == pytest.approx(1 / 3)


def test_k_cuts_the_list():
    got = [chunk("X", 1), chunk("X", 2), chunk("D", 3)]
    assert hit_at_k(got, MULTI, 2) == 0.0
    assert hit_at_k(got, MULTI, 3) == 1.0
    assert reciprocal_rank(got, MULTI, 2) == 0.0


def test_empty_gold_raises():
    with pytest.raises(ValueError):
        hit_at_k([chunk("D", 3)], [], 4)


from rag.evaluate import JUDGE_RUBRIC, build_judge_messages, parse_score  # noqa: E402


def test_judge_messages_layout():
    msgs = build_judge_messages("Why?", "Because A.", "Due to B.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == JUDGE_RUBRIC
    user = msgs[1]["content"]
    assert "Question: Why?" in user
    assert "Reference answer: Because A." in user
    assert user.endswith("Answer to grade: Due to B.")


def test_parse_score_valid():
    assert parse_score("SCORE: 2\nREASON: matches") == 2
    assert parse_score("SCORE:1 because partial") == 1
    assert parse_score("blah\nSCORE: 0") == 0


@pytest.mark.parametrize("bad", ["SCORE: 5", "SCORE: 12", "score: 2", "no score here", ""])
def test_parse_score_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_score(bad)


from rag.evaluate import (  # noqa: E402
    FAITHFULNESS_RUBRIC,
    build_faithfulness_messages,
    judge_faithfulness,
)
from rag.generate import REFUSAL  # noqa: E402

CHUNKS = [{"doc_id": "A", "page": 2, "text": "first"}, {"doc_id": "B", "page": 9, "text": "second"}]


def test_faithfulness_messages_show_labelled_passages_and_answer():
    msgs = build_faithfulness_messages("It is 5 [1].", CHUNKS)
    assert msgs[0]["content"] == FAITHFULNESS_RUBRIC
    user = msgs[1]["content"]
    assert "[1] (A, p.2)\nfirst" in user
    assert "[2] (B, p.9)\nsecond" in user
    assert user.endswith("Answer to check: It is 5 [1].")
    assert user.index("Passages:") < user.index("Answer to check:")


def test_faithfulness_skips_refusal_without_calling_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called for a refusal")

    monkeypatch.setattr("rag.evaluate.chat", boom)
    assert judge_faithfulness(REFUSAL, CHUNKS) is None


from rag.evaluate import abstention_report, is_abstention  # noqa: E402


def test_is_abstention_exact_sentence_only():
    assert is_abstention(REFUSAL)
    assert is_abstention(f"  {REFUSAL}\n")
    assert not is_abstention("The documents do not specify the range [4].")
    assert not is_abstention("The answer is 5 [1].")


def row(kind: str, abstained: bool) -> dict:
    return {"type": kind, "abstained": abstained}


def test_abstention_report_counts():
    rows = [
        row("factual", False), row("factual", True), row("multi", False), row("multi", False),
        row("unanswerable", True), row("unanswerable", True), row("unanswerable", False),
    ]
    rep = abstention_report(rows)
    assert rep["n_answerable"] == 4 and rep["n_unanswerable"] == 3
    assert rep["over_refusal"] == 0.25
    assert rep["correct_abstention"] == pytest.approx(2 / 3)
    assert rep["accuracy"] == pytest.approx(5 / 7)


def test_always_refusing_is_exposed_by_over_refusal():
    rows = [row("factual", True)] * 3 + [row("unanswerable", True)] * 2
    rep = abstention_report(rows)
    assert rep["correct_abstention"] == 1.0
    assert rep["over_refusal"] == 1.0


def test_empty_group_gives_none():
    rep = abstention_report([row("factual", False)])
    assert rep["correct_abstention"] is None


import csv  # noqa: E402
import json  # noqa: E402

from rag import evaluate  # noqa: E402
from rag.evaluate import evaluate_item, summarize, write_results  # noqa: E402

GOLD_ITEM = {
    "id": "f99", "type": "factual", "question": "Q?", "answer": "A.",
    "gold": [{"doc_id": "D", "pages": [3]}],
}
UNANS_ITEM = {"id": "u99", "type": "unanswerable", "question": "Q?", "answer": "-", "gold": []}


def fake_pipeline(monkeypatch, answer_text: str, chunks: list[dict]) -> None:
    monkeypatch.setattr(evaluate, "retrieve", lambda q, k, size, overlap, mode: chunks)
    monkeypatch.setattr(
        evaluate, "answer_from_chunks", lambda q, c: {"answer": answer_text, "chunks": c}
    )
    monkeypatch.setattr(evaluate, "judge_correctness", lambda q, g, a: {"score": 2, "reason": "ok"})
    monkeypatch.setattr(
        evaluate, "judge_faithfulness", lambda a, c: {"score": 1, "reason": "partly"}
    )


def test_evaluate_item_answered_gets_all_scores(monkeypatch):
    fake_pipeline(monkeypatch, "It is 5 [1].", [chunk("X", 1), chunk("D", 3)])
    row = evaluate_item(GOLD_ITEM, k=4, chunk_size=1000, chunk_overlap=100, mode="dense")
    assert (row["hit"], row["recall"], row["rr"]) == (1.0, 1.0, 0.5)
    assert row["correctness"] == 2 and row["faithfulness"] == 1
    assert row["abstained"] is False and row["judge_errors"] == []


def test_evaluate_item_refused_answerable_is_zero_without_judges(monkeypatch):
    fake_pipeline(monkeypatch, REFUSAL, [chunk("X", 1)])
    monkeypatch.setattr(evaluate, "judge_correctness", lambda *a: 1 / 0)
    row = evaluate_item(GOLD_ITEM, k=4, chunk_size=1000, chunk_overlap=100, mode="dense")
    assert row["abstained"] is True
    assert row["correctness"] == 0 and row["faithfulness"] is None
    assert row["hit"] == 0.0


def test_evaluate_item_unanswerable_has_no_retrieval_or_correctness(monkeypatch):
    fake_pipeline(monkeypatch, "Made up [1].", [chunk("X", 1)])
    row = evaluate_item(UNANS_ITEM, k=4, chunk_size=1000, chunk_overlap=100, mode="dense")
    assert row["hit"] is None and row["correctness"] is None
    assert row["faithfulness"] == 1


def test_malformed_judge_reply_is_recorded_not_raised(monkeypatch):
    fake_pipeline(monkeypatch, "It is 5 [1].", [chunk("D", 3)])

    def bad(*args):
        raise ValueError("no valid SCORE")

    monkeypatch.setattr(evaluate, "judge_correctness", bad)
    row = evaluate_item(GOLD_ITEM, k=4, chunk_size=1000, chunk_overlap=100, mode="dense")
    assert row["correctness"] is None and len(row["judge_errors"]) == 1
    assert summarize([row])["judge_errors"] == 1


def _row(kind, hit, rr, correctness, faith, abstained):
    return {
        "type": kind, "hit": hit, "recall": hit, "rr": rr, "correctness": correctness,
        "faithfulness": faith, "abstained": abstained, "judge_errors": [],
    }


def test_summarize_uses_only_applicable_rows():
    rows = [
        _row("factual", 1.0, 1.0, 2, 2, False),
        _row("factual", 0.0, 0.0, 0, None, True),
        _row("unanswerable", None, None, None, None, True),
    ]
    s = summarize(rows)
    assert s["hit"] == 0.5 and s["mrr"] == 0.5
    assert s["correct_mean"] == 1.0 and s["correct_full"] == 0.5
    assert s["faith_mean"] == 2.0 and s["faith_n"] == 1
    assert s["correct_abstention"] == 1.0 and s["over_refusal"] == 0.5


def test_write_results_upserts_summary_row(tmp_path):
    rows = [_row("factual", 1.0, 1.0, 2, 2, False)]
    config = {"chunk_size": 1000, "overlap": 100, "k": 4, "mode": "dense"}
    write_results("cfgA", config, rows, tmp_path)
    write_results("cfgB", config, rows, tmp_path)
    write_results("cfgA", config, rows, tmp_path)
    table = list(csv.DictReader(open(tmp_path / "summary.csv", encoding="utf-8")))
    assert sorted(r["name"] for r in table) == ["cfgA", "cfgB"]
    assert table[0]["hit"] == "1.0"
    lines = (tmp_path / "cfgA.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["type"] == "factual"


def test_no_retrieval_mode_skips_retrieval_and_context_metrics(monkeypatch):
    from rag import evaluate

    def boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(evaluate, "retrieve", boom)
    monkeypatch.setattr(evaluate, "judge_faithfulness", boom)
    monkeypatch.setattr(
        evaluate, "answer_no_retrieval", lambda q: {"answer": "Some answer", "chunks": []}
    )
    monkeypatch.setattr(evaluate, "judge_correctness", lambda q, g, a: {"score": 1, "reason": "x"})
    gold = [{"doc_id": "d", "pages": [1]}]
    item = {"id": "q1", "type": "factual", "question": "Q?", "answer": "A", "gold": gold}
    row = evaluate.evaluate_item(item, 0, 1000, 100, "none")
    assert row["retrieved"] == []
    assert row["hit"] is None and row["recall"] is None and row["rr"] is None
    assert row["correctness"] == 1 and row["faithfulness"] is None
