import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")

MIN_WRAPPED_LINE = 40
_SENTENCE_END = (".", "!", "?", ":")


_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

EDGE_LINES = 2
MIN_REPEAT_SHARE = 0.3
MIN_PAGES_FOR_HEADERS = 6
MAX_HEADER_LEN = 100
_PAGE_NUMBER = re.compile(r"\d{1,4}")


_REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d\d\b")
_CITATION_WORDS = re.compile(
    r"et al|arXiv|doi|Proceedings|Conference|Journal|pp\.|pages|Association|Linguistics"
    r"|preprint|In:|Vol\.|volume",
    re.IGNORECASE,
)
MIN_CITATION_DENSITY = 0.30


def strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


def normalize_ligatures(text: str) -> str:
    for ligature, letters in _LIGATURES.items():
        text = text.replace(ligature, letters)
    return text


def fix_hyphenation(text: str) -> str:
    return _HYPHEN_BREAK.sub("", text)


def join_line_breaks(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        is_last = i == len(lines) - 1
        stripped = line.strip()
        wrapped = (
            len(stripped) >= MIN_WRAPPED_LINE
            and not stripped.endswith(_SENTENCE_END)
            and not _NUMBERED_HEADING.match(stripped)
            and not (not is_last and _NUMBERED_HEADING.match(lines[i + 1].strip()))
        )
        out.append(" " if wrapped and not is_last else "\n")
    return "".join(out).rstrip("\n")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _edge_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def find_headers_footers(page_texts: list[str]) -> set[str]:
    if len(page_texts) < MIN_PAGES_FOR_HEADERS:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        for line in set(_edge_lines(text)):
            counts[line] = counts.get(line, 0) + 1
    threshold = MIN_REPEAT_SHARE * len(page_texts)
    return {ln for ln, n in counts.items() if n >= threshold and len(ln) <= MAX_HEADER_LEN}


def remove_headers_footers(page_texts: list[str]) -> list[str]:
    repeated = find_headers_footers(page_texts)
    cleaned: list[str] = []
    for text in page_texts:
        lines = text.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(non_empty[:EDGE_LINES] + non_empty[-EDGE_LINES:])
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (i in edge and (ln.strip() in repeated or _PAGE_NUMBER.fullmatch(ln.strip())))
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def citation_density(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _YEAR.search(ln) or _CITATION_WORDS.search(ln))
    return hits / len(lines)


def drop_references(page_texts: list[str]) -> list[str]:
    heading_page = heading_line = None
    for p, text in enumerate(page_texts):
        for i, line in enumerate(text.split("\n")):
            if _REFERENCES_HEADING.match(line):
                heading_page, heading_line = p, i
                break
        if heading_page is not None:
            break
    if heading_page is None:
        return page_texts

    result = list(page_texts)
    result[heading_page] = "\n".join(page_texts[heading_page].split("\n")[:heading_line])
    p = heading_page + 1
    while p < len(page_texts) and citation_density(page_texts[p]) >= MIN_CITATION_DENSITY:
        result[p] = ""
        p += 1
    return result


def clean_page(text: str) -> str:
    text = strip_control_chars(text)
    text = normalize_ligatures(text)
    text = fix_hyphenation(text)
    text = join_line_breaks(text)
    return normalize_whitespace(text)
