#!/usr/bin/env python3
"""Reproducibly acquire Sheth's original DDSA edition, following its page links.

The modern ISJS translation is a different, incomplete edition. Never silently
substitute it for the 1923–1928 Prakrit–Hindi dictionary. Acquisition is read-only
with respect to installed forms; parser/installation gates are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

URL = "https://dsal.uchicago.edu/cgi-bin/app/sheth_query.py?page={}"
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE = ROOT / "tmp/sheth-ddsa-20260911"
# DDSA's actual navigation stops at 952 with hrāsa. The linked first-edition
# scans end at printed 1278 including a supplement. These page systems MUST NOT
# be conflated: digital page numbers remain website locators until reconciled.
LAST_PAGE = 952


def fetch(cache: Path, limit: int | None = None) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    pages = []
    page = 1
    while True:
        path = cache / f"{page:04d}.html"
        cached = path.exists()
        if path.exists():
            raw = path.read_bytes()
        else:
            for attempt in range(4):
                try:
                    req = urllib.request.Request(URL.format(page), headers={
                        "User-Agent": "Jambu lexical research (https://github.com/moli-mandala)"
                    })
                    with urllib.request.urlopen(req, timeout=45) as response:
                        raw = response.read()
                    if b"<hw>" not in raw and b"Please click next to see digital content for this page." not in raw:
                        raise ValueError(f"Page {page}: no headwords; refusing an incomplete snapshot")
                    temp = path.with_suffix(".html.tmp")
                    temp.write_bytes(raw)
                    temp.replace(path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        text = raw.decode("utf-8")
        assert "<hw>" in text or "Please click next to see digital content for this page." in text, f"Invalid cached page {page}"
        pages.append({"page": page, "url": URL.format(page),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "headwords": text.count("<hw>"),
                      "status": "entries" if "<hw>" in text else "no-starting-headword-review-required"})
        links = {int(n) for n in re.findall(r"sheth_query\.py\?page=(\d+)", text)}
        following = sorted(n for n in links if n > page)
        complete = page == LAST_PAGE
        if complete and "hrāsa" not in text:
            raise ValueError("Final DDSA page lacks the verified final headword hrāsa")
        if not complete and not following:
            raise ValueError(f"Pagination stops at {page}, before verified digital end {LAST_PAGE}")
        if page % 25 == 0 or complete:
            print(f"page={page} headwords={sum(p['headwords'] for p in pages)}", flush=True)
        manifest = {"source": "sheth1923", "snapshot_date": "2026-09-11",
                    "edition": "1923–1928, DDSA data updated May 2026",
                    "edition_reconciliation": "pending: web pagination differs from linked printed scans",
                    "complete_scope": "DDSA web navigation only",
                    "complete": complete, "pages": pages,
                    "headwords": sum(p["headwords"] for p in pages)}
        temp = cache / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temp.replace(cache / "manifest.json")
        if complete or (limit is not None and len(pages) >= limit):
            return manifest
        assert following[0] == page + 1, f"Unexpected pagination after {page}: {following}"
        page = following[0]
        if not cached:
            time.sleep(0.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = fetch(args.cache, args.limit)
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
