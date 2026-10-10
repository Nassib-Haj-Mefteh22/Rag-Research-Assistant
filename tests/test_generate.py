from rag import generate
from rag.generate import REFUSAL, build_messages, extract_citations, format_context, strip_thinking

CHUNKS = [
    {"doc_id": "2501.00001", "page": 3, "text": "first text"},
    {"doc_id": "2502.00002", "page": 7, "text": "second text"},
]


def test_context_labels_in_rank_order():
    ctx = format_context(CHUNKS)
    assert ctx.startswith("[1] (2501.00001, p.3)\nfirst text")
    assert "[2] (2502.00002, p.7)\nsecond text" in ctx
    assert ctx.index("[1]") < ctx.index("[2]")


def test_messages_structure():
    msgs = build_messages("Why?", CHUNKS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert REFUSAL in msgs[0]["content"]
    assert msgs[1]["content"].endswith("Question: Why?")
    assert "first text" in msgs[1]["content"]


def test_extract_citations_valid_invalid_duplicates():
    valid, invalid = extract_citations("A [1]. B [2][1]. C [7]. D [0].", 3)
    assert valid == [1, 2]
    assert invalid == [0, 7]


def test_strip_thinking():
    assert strip_thinking("<think>hmm [9]\nmore</think>\nAnswer [1]") == "Answer [1]"
    assert strip_thinking("plain") == "plain"


def test_answer_with_fakes(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda q, k, mode: CHUNKS)
    monkeypatch.setattr(generate, "chat", lambda *a, **kw: "<think>x</think>Yes [2] and [5].")
    monkeypatch.setattr(generate, "load_settings", lambda: type("S", (), {"gen_model": "m"})())
    result = generate.answer("q")
    assert result["answer"] == "Yes [2] and [5]."
    assert result["sources"] == [("2502.00002", 7)]
    assert result["invalid_citations"] == [5]
    assert result["chunks"] == CHUNKS
    assert not result["is_refusal"]


def test_refusal_flag(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda q, k, mode: CHUNKS)
    monkeypatch.setattr(generate, "chat", lambda *a, **kw: REFUSAL)
    monkeypatch.setattr(generate, "load_settings", lambda: type("S", (), {"gen_model": "m"})())
    assert generate.answer("q")["is_refusal"]
