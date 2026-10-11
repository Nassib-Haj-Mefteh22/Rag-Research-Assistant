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
USER_AGENT = "rag-assistant-portfolio/0.1 (student project; polite downloader)"
DELAY_SECONDS = 3.0
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}v\d+$")


def parse_ids(text: str) -> list[str]:
    ids: list[str] = []
    for line in text.splitlines():
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        if not ARXIV_ID.match(token):
            raise ValueError(f"Not a versioned arXiv id (e.g. 2610.01936v1): {token!r}")
        if token in ids:
            continue
        if base_id(token) in (base_id(i) for i in ids):
            raise ValueError(f"Two versions of the same paper: {token!r}")
        ids.append(token)
    return ids


def base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def download_one(client: httpx.Client, arxiv_id: str, dest: Path) -> None:
    response = client.get(PDF_URL.format(arxiv_id=arxiv_id))
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"{arxiv_id}: response is not a PDF")
    part = dest.with_suffix(".pdf.part")
    part.write_bytes(response.content)
    part.replace(dest)


def parse_entries(xml_text: str) -> list[dict]:
    records: list[dict] = []
    for entry in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
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
            dest = RAW_DIR / f"{base_id(arxiv_id)}.pdf"
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
        metadata = fetch_metadata(client, [base_id(i) for i in ids])
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
