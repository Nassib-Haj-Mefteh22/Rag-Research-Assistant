"""Step 2: download arXiv PDFs listed in corpus/corpus_ids.txt, write metadata.json."""

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from rag.config import ROOT

IDS_FILE = ROOT / "corpus" / "corpus_ids.txt"
METADATA_FILE = ROOT / "corpus" / "metadata.json"
RAW_DIR = ROOT / "data" / "raw"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
# WHY: arXiv asks automated clients to identify themselves and to go slowly.
USER_AGENT = "rag-assistant-portfolio/0.1 (student project; polite downloader)"
DELAY_SECONDS = 3.0
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


def parse_ids(text: str) -> list[str]:
    """Return arXiv IDs from the corpus file, ignoring comments, blanks and duplicates.

    WHY a pure function: the file format (`id  # title`) is the one thing that can break
    silently, so it is parsed in one place and unit-tested without touching the network.
    """
    ids: list[str] = []
    for line in text.splitlines():
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        if not ARXIV_ID.match(token):
            raise ValueError(f"Not an arXiv id: {token!r}")
        if token not in ids:
            ids.append(token)
    return ids


def download_one(client: httpx.Client, arxiv_id: str, dest: Path) -> None:
    """Fetch one PDF to `dest`. Writes to a .part file first so a crash never leaves a broken PDF.

    WHY the %PDF check: arXiv can answer 200 with an HTML error page; saving that as .pdf
    would only fail later, at parsing time, far from the cause.
    """
    response = client.get(PDF_URL.format(arxiv_id=arxiv_id))
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"{arxiv_id}: response is not a PDF")
    part = dest.with_suffix(".pdf.part")
    part.write_bytes(response.content)
    part.replace(dest)


def parse_entries(xml_text: str) -> list[dict]:
    """Turn an arXiv Atom response into metadata records.

    WHY pure and separate from the HTTP call: the XML shape is the fragile part, so it is
    unit-tested on a small sample. Titles are whitespace-normalised because the API wraps
    long titles over several lines. Year = first-version `published` date.
    """
    records: list[dict] = []
    for entry in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        # <id> looks like http://arxiv.org/abs/2610.01936v1: keep the id, drop the version.
        arxiv_id = re.sub(r"v\d+$", "", entry.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1])
        records.append(
            {
                "id": arxiv_id,
                "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
                "authors": [
                    " ".join(a.findtext(f"{ATOM}name", "").split())
                    for a in entry.findall(f"{ATOM}author")
                ],
                "year": int(entry.findtext(f"{ATOM}published", "")[:4]),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return records


def fetch_metadata(client: httpx.Client, ids: list[str]) -> list[dict]:
    """One API call for all ids (50 fit in one request), returned in corpus-file order."""
    response = client.get(API_URL, params={"id_list": ",".join(ids), "max_results": len(ids)})
    response.raise_for_status()
    by_id = {r["id"]: r for r in parse_entries(response.text)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"arXiv API returned no entry for: {missing}")
    return [by_id[i] for i in ids]


def main() -> int:
    ids = parse_ids(IDS_FILE.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60.0
    ) as client:
        for arxiv_id in ids:
            dest = RAW_DIR / f"{arxiv_id}.pdf"
            if dest.exists():
                skipped.append(arxiv_id)
                continue
            try:
                download_one(client, arxiv_id, dest)
                downloaded.append(arxiv_id)
            except (httpx.HTTPError, ValueError) as err:
                failed.append(arxiv_id)
                print(f"FAILED {arxiv_id}: {err}")
            time.sleep(DELAY_SECONDS)
        metadata = fetch_metadata(client, ids)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metadata entries={len(metadata)} -> {METADATA_FILE.relative_to(ROOT)}")
    print(
        f"ids={len(ids)} downloaded={len(downloaded)} "
        f"skipped={len(skipped)} failed={len(failed)}"
    )
    print("samples downloaded:", downloaded[:5])
    print("samples skipped:", skipped[:5])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
