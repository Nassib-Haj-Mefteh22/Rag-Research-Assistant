from rag.clean import (
    citation_density,
    clean_page,
    drop_references,
    find_headers_footers,
    fix_hyphenation,
    join_line_breaks,
    normalize_ligatures,
    normalize_whitespace,
    remove_headers_footers,
    strip_control_chars,
)


def test_strip_control_chars_keeps_newline_and_tab():
    assert strip_control_chars("a\x01b\x10c\nd\te") == "abc\nd\te"


def test_normalize_ligatures():
    assert normalize_ligatures("ﬁne-tuning, eﬃcient") == "fine-tuning, efficient"


def test_ligatures_do_not_touch_superscripts():
    assert normalize_ligatures("x²") == "x²"


def test_fix_hyphenation_joins_split_word():
    assert fix_hyphenation("a knowl-\nedge graph") == "a knowledge graph"


def test_fix_hyphenation_leaves_uppercase_and_digits():
    assert fix_hyphenation("GPT-\n4 and BM-\n25") == "GPT-\n4 and BM-\n25"


def test_fix_hyphenation_leaves_inline_hyphen():
    assert fix_hyphenation("state-of-the-art") == "state-of-the-art"


def test_join_line_breaks_joins_wrapped_paragraph():
    text = "The retriever returns the top documents and\nthe generator reads them."
    assert join_line_breaks(text) == (
        "The retriever returns the top documents and the generator reads them."
    )


def test_join_line_breaks_keeps_short_lines_separate():
    text = "Question\nType\nRetrieval"
    assert join_line_breaks(text) == "Question\nType\nRetrieval"


def test_join_line_breaks_keeps_sentence_end():
    text = "This is a complete and fairly long sentence here.\nNew paragraph starts."
    assert join_line_breaks(text) == text


def test_join_line_breaks_keeps_numbered_heading_apart():
    text = (
        "5.6 Post-submission external-scaffold validation\n"
        "All eight OpenHands entries met the pre-verdict source threshold, as shown."
    )
    assert join_line_breaks(text) == text


def test_join_line_breaks_does_not_glue_heading_to_previous_paragraph():
    text = (
        "the end of a paragraph that wraps and has no final period\n"
        "3 Methodology and design of the study overall\n"
        "Body text."
    )
    assert join_line_breaks(text) == text


def _page(header: str, body: str) -> str:
    return f"{header}\nfirst body line {body}\nsecond body line {body}\nthird {body}\nlast {body}"


def test_find_headers_footers_detects_repeated_edge_line():
    pages = [_page("Conference 2017", str(i)) for i in range(10)]
    assert find_headers_footers(pages) == {"Conference 2017"}


def test_find_headers_footers_ignores_repeated_line_in_the_middle():
    pages = [f"top {i}\na{i}\nQwen\nb{i}\nc{i}\nd{i}\nbottom {i}" for i in range(10)]
    assert find_headers_footers(pages) == set()


def test_find_headers_footers_skips_short_documents():
    pages = [_page("Conference 2017", str(i)) for i in range(3)]
    assert find_headers_footers(pages) == set()


def test_remove_headers_footers_keeps_middle_copy_of_same_text():
    pages = [
        f"Conference 2017\nx{i}\nConference 2017\ny{i}\nz{i}\nw{i}\nend {i}" for i in range(10)
    ]
    out = remove_headers_footers(pages)
    assert out[0] == "x0\nConference 2017\ny0\nz0\nw0\nend 0"


def test_remove_headers_footers_drops_bare_page_number_even_in_short_doc():
    pages = ["intro line\nbody one\nbody two\nbody three\n4", "next page\nbody\nmore\nend\n5"]
    out = remove_headers_footers(pages)
    assert out == ["intro line\nbody one\nbody two\nbody three", "next page\nbody\nmore\nend"]


def test_remove_headers_footers_keeps_number_in_the_middle():
    page = "title\nresults\n42\nmore text\nlast line here\nfinal words"
    assert remove_headers_footers([page]) == [page]


def test_normalize_whitespace_handles_nbsp():
    assert normalize_whitespace("Query:\xa0\xa0What is") == "Query: What is"


def test_normalize_whitespace():
    assert normalize_whitespace("a   b \t c \n\n\n\n d  ") == "a b c\n\nd"


def test_clean_page_order_hyphen_before_join():
    text = "We build a very large knowl-\nedge graph from the documents we have."
    assert clean_page(text) == "We build a very large knowledge graph from the documents we have."


CITATION = "[1] Lee et al., 2025. Dense retrieval for QA. In Proceedings of ACL."
BODY = "We train the model for three epochs with a learning rate of one e-4."


def test_citation_density_high_for_references_low_for_body():
    assert citation_density("\n".join([CITATION] * 5)) == 1.0
    assert citation_density("\n".join([BODY] * 5)) == 0.0
    assert citation_density("") == 0.0


def test_drop_references_keeps_appendix_after_bibliography():
    pages = [
        BODY,
        f"{BODY}\nReferences\n{CITATION}\n{CITATION}",
        "\n".join([CITATION] * 4),
        "A Prompts\n" + "\n".join([BODY] * 4),
    ]
    assert drop_references(pages) == [BODY, BODY, "", pages[3]]


def test_drop_references_bibliography_to_end_of_doc():
    pages = [BODY, f"References\n{CITATION}", "\n".join([CITATION] * 4)]
    assert drop_references(pages) == [BODY, "", ""]


def test_drop_references_without_heading_changes_nothing():
    pages = [BODY, BODY]
    assert drop_references(pages) == pages


def test_drop_references_keeps_length_so_page_numbers_stay_aligned():
    pages = [BODY, f"References\n{CITATION}", "\n".join([CITATION] * 4), BODY]
    assert len(drop_references(pages)) == len(pages)


def test_drop_references_heading_is_whole_line_only():
    pages = [f"See the references in Section 2 for details.\n{BODY}", BODY]
    assert drop_references(pages) == pages


def test_citation_density_counts_wrapped_citation_split_over_lines():
    wrapped = (
        "Jon Saad-Falcon, Omar Khattab, and\n"
        "Matei Zaharia. 2024. ARES: An automated\n"
        "evaluation framework. In Proceedings of\n"
        "the North American Chapter of the\n"
        "Association for Computational Linguistics"
    )
    assert citation_density(wrapped) >= 0.30
