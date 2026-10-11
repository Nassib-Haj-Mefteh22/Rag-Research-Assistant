from rag.chunk import chunk_pages, chunk_stats

SENTENCE = "Retrieval augmented generation grounds answers in documents. "
PAGES = [
    {"doc_id": "d1", "page": 1, "text": SENTENCE * 20},
    {"doc_id": "d1", "page": 3, "text": "Short page."},
    {"doc_id": "d2", "page": 2, "text": SENTENCE * 12},
]


def test_no_empty_chunks_and_size_respected():
    chunks = chunk_pages(PAGES, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(c["text"].strip() for c in chunks)
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_metadata_kept_and_ids_unique():
    chunks = chunk_pages(PAGES, chunk_size=200, chunk_overlap=20)
    assert {(c["doc_id"], c["page"]) for c in chunks} == {("d1", 1), ("d1", 3), ("d2", 2)}
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "d1_p1_c0"
    short = [c for c in chunks if c["page"] == 3]
    assert len(short) == 1 and short[0]["text"] == "Short page."


def test_consecutive_chunks_overlap():
    chunks = [c for c in chunk_pages(PAGES, 200, 40) if c["doc_id"] == "d1" and c["page"] == 1]
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt["text"][:20].lstrip(". ") in prev["text"][-60:]


def test_whitespace_page_gives_no_chunks():
    assert chunk_pages([{"doc_id": "d", "page": 1, "text": "   \n\n  "}], 200, 20) == []


def test_chunk_stats_on_known_lengths():
    chunks = [{"text": "a" * n} for n in (10, 20, 300)]
    stats = chunk_stats(chunks, tiny=100)
    assert stats == {"count": 3, "mean": 110, "median": 20, "min": 10, "max": 300, "tiny": 2}
