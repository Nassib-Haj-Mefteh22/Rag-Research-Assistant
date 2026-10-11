from rag.parse import process_doc

CITATION = "[1] Lee et al., 2025. Dense retrieval for QA. In Proceedings of ACL."
HEADER = "Conference 2017"


def _body(i: int) -> str:
    return f"{HEADER}\nWe describe experiment {i} in detail here and\nwe report result {i}.\n{i}"


def test_process_doc_drops_references_keeps_appendix_and_original_page_numbers():
    pages = [_body(i) for i in range(1, 7)]
    pages.append("References\n" + "\n".join([CITATION] * 5))
    pages.append("\n".join([CITATION] * 5))
    pages.append("A Prompts\nThe judge prompt we used for all runs is shown below.")
    out = dict(process_doc(pages))
    assert sorted(out) == [1, 2, 3, 4, 5, 6, 9]
    assert out[1] == "We describe experiment 1 in detail here and we report result 1."
    assert "Conference 2017" not in out[1]
    assert out[9].startswith("A Prompts")


def test_process_doc_drops_pages_that_become_empty():
    pages = [_body(i) for i in range(1, 7)] + ["\x01\n42"]
    assert 7 not in dict(process_doc(pages))
