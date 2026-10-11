import pytest

from rag.download import base_id, parse_entries, parse_ids

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2610.01936v1</id>
    <published>2026-10-01T12:00:00Z</published>
    <title>Mapping the RAG
  Landscape</title>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan  Turing</name></author>
  </entry>
</feed>"""


def test_parse_ids_strips_comments_and_blanks() -> None:
    text = "# header\n\n2610.01936v1  # Some title\n2609.38473v2\n"
    assert parse_ids(text) == ["2610.01936v1", "2609.38473v2"]


def test_parse_ids_drops_duplicates() -> None:
    assert parse_ids("2610.01936v1\n2610.01936v1  # again\n") == ["2610.01936v1"]


def test_parse_ids_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_ids("not-an-id\n")


def test_parse_ids_requires_a_version() -> None:
    with pytest.raises(ValueError):
        parse_ids("2610.01936\n")


def test_parse_ids_rejects_two_versions_of_one_paper() -> None:
    with pytest.raises(ValueError):
        parse_ids("2610.01936v1\n2610.01936v2\n")


def test_base_id_drops_the_version() -> None:
    assert base_id("2610.01936v1") == "2610.01936"
    assert base_id("2609.06444v3") == "2609.06444"


def test_parse_entries_normalises_fields() -> None:
    (record,) = parse_entries(SAMPLE_XML)
    assert record == {
        "id": "2610.01936",
        "title": "Mapping the RAG Landscape",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "year": 2026,
        "url": "https://arxiv.org/abs/2610.01936",
    }
