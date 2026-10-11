from rag.cli import build_parser, format_sources

TITLES = {"2501.00001": "Paper One", "2502.00002": "Paper Two"}


def test_sources_are_deduplicated_in_first_cited_order():
    sources = [("2502.00002", 7), ("2501.00001", 3), ("2502.00002", 7)]
    assert format_sources(sources, TITLES) == [
        "[2502.00002] Paper Two, p.7",
        "[2501.00001] Paper One, p.3",
    ]


def test_same_doc_different_pages_are_kept():
    lines = format_sources([("2501.00001", 3), ("2501.00001", 4)], TITLES)
    assert len(lines) == 2


def test_unknown_doc_id_does_not_crash():
    assert format_sources([("9999.99999", 1)], TITLES) == ["[9999.99999] (title unknown), p.1"]


def test_no_sources_gives_empty_list():
    assert format_sources([], TITLES) == []


def test_parser_defaults_to_hybrid_rerank_k4():
    args = build_parser().parse_args(["ask", "What is RAG?"])
    assert (args.mode, args.k, args.show_chunks) == ("hybrid+rerank", 4, False)
