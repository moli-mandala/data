#!/usr/bin/env python3
"""Snapshot Lāḷas DDSA second edition; user confirmed SARVA permission 2026-09-11.

Acquisition only. No canonical forms are changed by this command. Web locators
are distinct from the linked first-edition images until edition reconciliation.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
URL = 'https://dsal.uchicago.edu/cgi-bin/app/lalasa-2nd_query.py?page={}'
CONTINUATION = 'Please click next to see digital content for this page.'


def inspect_page(text, page):
    if '<hw>' not in text and CONTINUATION not in text:
        raise ValueError(f'Page {page} has neither headwords nor an explicit continuation')
    following = sorted({int(n) for n in re.findall(r'lalasa-2nd_query\.py\?page=(\d+)', text) if int(n) > page})
    if '<hw>' not in text and not following:
        raise ValueError(f'Empty terminal page {page} requires manual review')
    if following and following[0] != page + 1:
        raise ValueError(f'Non-sequential pagination at {page}: {following}')
    return following


def fetch(cache, limit=None):
    cache.mkdir(parents=True, exist_ok=True)
    pages = []
    page = 1
    while True:
        path = cache / f'{page:04d}.html'
        cached = path.exists()
        if not cached:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(URL.format(page), timeout=45) as response:
                        raw = response.read()
                    text = raw.decode('utf-8')
                    inspect_page(text, page)
                    temp = path.with_suffix('.tmp')
                    temp.write_bytes(raw)
                    temp.replace(path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        following = inspect_page(text, page)
        pages.append(dict(page=page, url=URL.format(page),
                          sha256=hashlib.sha256(raw).hexdigest(), headwords=text.count('<hw>'),
                          status='entries' if '<hw>' in text else 'continuation-review-required'))
        complete = not following
        manifest = dict(source='lalasa2013', snapshot_date='2026-09-11',
                        edition='Second edition 2013; DDSA revision November 2021',
                        permission='User confirmed permission via SARVA on 2026-09-11',
                        complete=complete, complete_scope='DDSA navigation only; printed edition reconciliation pending',
                        pages=pages, headwords=sum(p['headwords'] for p in pages))
        temp = cache / 'manifest.json.tmp'
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        temp.replace(cache / 'manifest.json')
        if page % 100 == 0 or complete:
            print(f"page={page} articles={manifest['headwords']} complete={complete}", flush=True)
        if complete or (limit and len(pages) >= limit):
            return manifest
        assert following[0] == page + 1, following
        page = following[0]
        if not cached:
            time.sleep(.15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'tmp/lalasa-ddsa-20260911')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    fetch(args.cache, args.limit)
