#!/usr/bin/env python3
"""Import Sanskrit-linked entries from the public Dictionary of Gāndhārī API.

The website exposes the same JSON endpoints used by its browser UI.  We first ask
for every article with a Sanskrit etymon, cache each article response, then attach
an article to CDIAL only when its Sanskrit form has one unique normalized CDIAL
head match.  Everything else is retained in an audit CSV for manual review.

Typical refresh (run from ``data/``)::

    uv run python data/other/forms/raw_data/gandhari_org.py

The cache makes interrupted and subsequent runs resumable.  ``--workers`` defaults
to four to avoid placing unnecessary load on gandhari.org.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = "https://gandhari.org/main/plugins/dev/php"
SEARCH_URL = f"{BASE}/searchLoader.php"
LEMMA_URL = f"{BASE}/lemmaLoader.php"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala)"
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "data/other/forms/20260805-gandhari-org.csv"
DEFAULT_AUDIT = ROOT / "tmp/gandhari-org-audit.csv"
DEFAULT_CACHE = ROOT / "tmp/gandhari-org-cache"
DEFAULT_CDIAL = ROOT / "data/cdial/params.csv"


def strip_html(value: str | None) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", "; ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_sanskrit(value: str) -> str:
    """A strict comparison key: discard accents/notation, retain segmental marks."""
    value = html.unescape(value).casefold().strip()
    value = value.replace("ṁ", "ṃ").replace("m̐", "ṃ")
    value = re.sub(r"^(?:cf\.?|or|also)\s+", "", value)
    value = re.sub(r"^[*†‡√]+|[-‐‑–—*†‡?¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", value)
    # CDIAL prints Vedic accents on otherwise identical dictionary heads.
    chars = []
    accent_marks = {"\u0300", "\u0301", "\u0302", "\u0340", "\u0341", "\u0951", "\u0952"}
    for char in unicodedata.normalize("NFD", value):
        if char in accent_marks:
            continue
        chars.append(char)
    value = unicodedata.normalize("NFC", "".join(chars))
    return re.sub(r"[\s.·'’(){}\[\]]+", "", value)


def extract_sanskrit_etyma(raw: str | None) -> list[str]:
    """Extract the Sanskrit portion of Gandhari.org's compact etymology HTML."""
    raw = raw or ""
    markers = list(re.finditer(r"\bSkt\.?\s*", raw, flags=re.I))
    values = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        segment = raw[marker.end() : end]
        pali = re.search(r"(?:,|;)\s*P(?:\.|āli\b|\s+)", segment, flags=re.I)
        if pali:
            segment = segment[: pali.start()]
        segment = segment.rsplit(")", 1)[0]
        italics = re.findall(r"<i[^>]*>(.*?)</i>", segment, flags=re.I | re.S)
        text = ", ".join(strip_html(part) for part in italics) if italics else strip_html(segment)
        for value in re.split(r"\s*(?:,|;|/|\bor\b)\s*", text):
            if normalize_sanskrit(value) and value not in values:
                values.append(value)
    return values


def cdial_index(path: Path = DEFAULT_CDIAL) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            cdial_id, heads = row[:2]
            for head in re.split(r"\s*,\s*", heads):
                key = normalize_sanskrit(head)
                if key:
                    pair = (cdial_id, head)
                    if pair not in index.setdefault(key, []):
                        index[key].append(pair)
    return index


def match_cdial(etyma: list[str], index: dict[str, list[tuple[str, str]]]):
    matches = {
        pair
        for etymon in etyma
        for pair in index.get(normalize_sanskrit(etymon), ())
    }
    ids = {pair[0] for pair in matches}
    if len(ids) == 1:
        cdial_id = next(iter(ids))
        return cdial_id, sorted(head for ident, head in matches if ident == cdial_id), "matched"
    if ids:
        return "", sorted(f"{ident}:{head}" for ident, head in matches), "ambiguous"
    return "", [], "unmatched"


def api_get(url: str, packet: dict, attempts: int = 5) -> dict:
    query = urllib.parse.urlencode({"db": "gandhari", "strJSON": json.dumps(packet)})
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:  # network/API errors are retried, then surfaced
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def sanskrit_entry_ids() -> list[str]:
    result = api_get(
        SEARCH_URL,
        {
            "arg": "lemmaSearch",
            "dictionary": "gd",
            "searchstring": "*",
            "searchtype": "S",
            "searchwave": 3,
            "page": "dictionary",
            "history": False,
        },
    )
    return sorted({str(row["id"]) for row in result["list"] if str(row["id"]).isdigit()}, key=int)


def fetch_entry(entry_id: str, cache_dir: Path) -> dict:
    cache_file = cache_dir / f"{entry_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    entry = api_get(
        LEMMA_URL,
        {"dictionary": "gd", "gd3id": int(entry_id), "phase": "main"},
    )
    cache_file.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return entry


def fetch_all(ids: list[str], cache_dir: Path, workers: int) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_entry, entry_id, cache_dir): entry_id for entry_id in ids}
        for done, future in enumerate(as_completed(futures), 1):
            entries.append(future.result())
            if done % 250 == 0 or done == len(ids):
                print(f"fetched {done}/{len(ids)} articles", flush=True)
    return sorted(entries, key=lambda entry: int(entry["_id"]))


def source_row(entry: dict, cdial_id: str, etyma: list[str]) -> list[str]:
    entry_id = str(entry["_id"])
    native = strip_html(entry.get("_lemNative"))
    parts = [f"Gandhari.org dictionary article {entry_id}"]
    if entry.get("_pos"):
        parts.append(f"part of speech: {strip_html(entry['_pos'])}")
    if entry.get("_morphology"):
        parts.append(f"morphology: {strip_html(entry['_morphology'])}")
    if entry.get("_citations"):
        parts.append(f"attestation: {strip_html(entry['_citations'])}")
    slug = urllib.parse.quote(entry["_lem"].lower(), safe="")
    notes = "; ".join(parts) + f"; https://gandhari.org/dictionary/{slug}"
    etymology = strip_html(entry.get("_etymologyDisp") or entry.get("_etymology"))
    if not etymology:
        etymology = "Skt. " + ", ".join(etyma)
    return [
        "Dhp", cdial_id, entry.get("_lem", ""), strip_html(entry.get("_def")), native,
        entry.get("_phonetic") or "", notes, "gandhari", "", etymology,
        f"gandhari:{entry_id}", "", "", "", "",
    ]


def build(entries: list[dict], output: Path, audit: Path, cdial_path: Path = DEFAULT_CDIAL):
    index = cdial_index(cdial_path)
    rows, audit_rows = [], []
    counts = {"matched": 0, "ambiguous": 0, "unmatched": 0, "missing": 0}
    for entry in entries:
        etyma = extract_sanskrit_etyma(entry.get("_etymology"))
        if not etyma:
            status, cdial_id, candidates = "missing", "", []
        else:
            cdial_id, candidates, status = match_cdial(etyma, index)
        counts[status] += 1
        if status == "matched":
            rows.append(source_row(entry, cdial_id, etyma))
        audit_rows.append([
            entry.get("_id", ""), entry.get("_lem", ""), " | ".join(etyma), status,
            cdial_id, " | ".join(candidates), strip_html(entry.get("_def")),
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Gandhari_ID", "Lemma", "Sanskrit", "Status", "CDIAL_ID", "Candidates", "Gloss"])
        writer.writerows(audit_rows)
    print(f"wrote {len(rows)} rows to {output}; audit counts: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cdial", type=Path, default=DEFAULT_CDIAL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ids", help="comma-separated article IDs (for testing)")
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else sanskrit_entry_ids()
    entries = fetch_all(ids, args.cache, max(1, args.workers))
    build(entries, args.output, args.audit, args.cdial)


if __name__ == "__main__":
    main()
