"""Step 3.3: cleaning rules applied to the raw text of one page.

WHY a separate module from parse.py: parse.py stays a raw extractor, so when a page looks
wrong we can tell whether the PDF or one of these rules caused it. Each rule is a small
pure function (str -> str) so it can be unit-tested alone.
"""

import re

# Control characters except \n (0x0a) and \t (0x09). They come from math/symbol fonts.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Explicit ligature map instead of NFKC: NFKC would also rewrite superscripts (x² -> x2)
# and math letters, which changes meaning in a scientific corpus.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# "knowl-\nedge": a lowercase letter, hyphen, newline, lowercase letter.
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")

# A line shorter than this is treated as a heading / table cell / list item, not as the
# middle of a wrapped paragraph. Tuned by eye on the corpus, not a derived constant.
MIN_WRAPPED_LINE = 40
_SENTENCE_END = (".", "!", "?", ":")


# Section headings like "5.6 Post-submission validation" or "3 Methodology": digits and
# dots, then a capital letter. Without this rule a long heading with no final period looks
# like a wrapped line and gets glued to the paragraph below it.
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

# Header/footer zone: a repeated line only counts if it is among the first/last lines of a
# page. Repetition alone also matches table cells in the middle of the page.
EDGE_LINES = 2
MIN_REPEAT_SHARE = 0.3
MIN_PAGES_FOR_HEADERS = 6
MAX_HEADER_LEN = 100
_PAGE_NUMBER = re.compile(r"\d{1,4}")


# References: a line with this heading starts the bibliography. Every paper in the corpus
# has exactly one (50/50), but many put an appendix AFTER it, so position alone is not enough.
_REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d\d\b")
_CITATION_WORDS = re.compile(
    r"et al|arXiv|doi|Proceedings|Conference|Journal|pp\.|pages|Association|Linguistics"
    r"|preprint|In:|Vol\.|volume",
    re.IGNORECASE,
)
# A page is "bibliography" if at least this share of its lines look like citations.
# Measured on the corpus: pages BEFORE the References heading score <= 0.28 for 99% of
# them, reference pages mostly 0.30-0.73, so 0.30 separates them (Step 3.4).
MIN_CITATION_DENSITY = 0.30


def strip_control_chars(text: str) -> str:
    """Remove non-printable characters that carry no text."""
    return _CONTROL_CHARS.sub("", text)


def normalize_ligatures(text: str) -> str:
    """Replace ligature characters (one glyph) with their letters so 'fine' matches 'fine'."""
    for ligature, letters in _LIGATURES.items():
        text = text.replace(ligature, letters)
    return text


def fix_hyphenation(text: str) -> str:
    """Glue words the PDF split at a line end: 'knowl-\\nedge' -> 'knowledge'.

    Only joins when the next line starts lowercase, so 'GPT-\\n4' is left alone. Known
    limit: a real compound split at a line end ('state-of-the-\\nart') loses its hyphen.
    """
    return _HYPHEN_BREAK.sub("", text)


def join_line_breaks(text: str) -> str:
    """Turn the PDF's width-driven line breaks into spaces, keeping real structure.

    A newline is replaced by a space only if the line before it is long and does not end
    a sentence. Short lines (headings, table cells, list items) keep their own line.
    Numbered headings are never joined, neither to the line after nor to the line before.
    This is a heuristic: it can miss a wrapped line that is very short.
    """
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
    """Collapse runs of spaces/tabs/non-breaking spaces, trim line ends, cap blank lines at one.

    \\xa0 (non-breaking space) is how PDFs often encode spaces; it carries no meaning here.
    """
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _edge_lines(text: str) -> list[str]:
    """Stripped non-empty lines in the first/last EDGE_LINES lines of a page."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def find_headers_footers(page_texts: list[str]) -> set[str]:
    """Lines that repeat at the top/bottom of at least 30% of one document's pages.

    Both conditions are required: repeated AND at the page edge. Short docs are skipped
    (too few pages to tell a header from a coincidence). Each page counts a line once.
    """
    if len(page_texts) < MIN_PAGES_FOR_HEADERS:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        for line in set(_edge_lines(text)):
            counts[line] = counts.get(line, 0) + 1
    threshold = MIN_REPEAT_SHARE * len(page_texts)
    return {ln for ln, n in counts.items() if n >= threshold and len(ln) <= MAX_HEADER_LEN}


def remove_headers_footers(page_texts: list[str]) -> list[str]:
    """Delete repeated edge lines (find_headers_footers) and bare page numbers.

    A page number changes on every page, so repetition cannot find it: an edge line that
    is only 1-4 digits is dropped on its own. Known risk: a table whose last row is a
    lone number loses that cell (rare, accepted).

    Runs on RAW text (before join_line_breaks), because it needs the original line
    structure. Only edge lines are removed, so a table cell with the same text in the
    middle of a page is kept.
    """
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
    """Share of non-empty lines containing a year OR a citation word.

    WHY content instead of position: appendices come after the bibliography in many
    papers, and their headings ('A Title' vs 'A. Title') are too varied to match, while a
    bibliography page is full of years and words like 'et al', 'arXiv', 'Proceedings'.
    WHY 'or', not 'and': one citation wraps over several lines, so its year and its
    'In Proceedings' usually sit on DIFFERENT lines; requiring both missed real pages.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _YEAR.search(ln) or _CITATION_WORDS.search(ln))
    return hits / len(lines)


def drop_references(page_texts: list[str]) -> list[str]:
    """Remove the bibliography of ONE document, keep any appendix after it.

    1. Find the page with the 'References' heading; keep the text above it, drop below.
    2. Drop the following pages while they look like a bibliography.
    3. At the first page that does not, stop dropping and keep everything to the end.
    The output has the SAME length as the input: dropped pages become "" instead of being
    removed, so index i is still page i+1 and citations keep the right page number. The
    caller (3.5) filters out the empty pages. No heading found -> nothing changes.
    Runs on RAW text, since it needs the original lines for the heading.
    """
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
    """Apply all text rules in a fixed order.

    Order matters: hyphenation must run BEFORE line joining. Otherwise the line break in
    'knowl-\\nedge' becomes a space and we get 'knowl- edge' that can no longer be fixed.
    """
    text = strip_control_chars(text)
    text = normalize_ligatures(text)
    text = fix_hyphenation(text)
    text = join_line_breaks(text)
    return normalize_whitespace(text)
