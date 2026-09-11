"""Reproduce the 406 curated donor heads for the approved Shinaic review (2026-09-10).

Input is the checked, source-attributed editorial audit, not a live scrape.
Only reviewed lexical heads are installed; full dictionary coverage is excluded.
The parameter format is intentional: these are curated donor etyma, not new
Shinaic attestations. Preserve source transcription, applying NFC only; raw
source variants and evidence remain in the audit. No sound correspondence is
inferred by this preservation route. No network, OCR, or fuzzy matching.
"""
import argparse
import csv
import io
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).with_name('20260910-shinaic-donors-audit.json')
OUTPUT = ROOT / 'data/other/params/20260910-shinaic-donors.csv'

def rows():
    records = json.loads(AUDIT.read_text())
    assert len(records) == 406
    assert len({r['ID'] for r in records}) == len(records)
    for r in records:
        assert r['Status'] == 'approved'
        assert r['Language_ID'] in {'Psht', 'H', 'Kho', 'Gaw'}
        assert r['Form'] and r['Gloss'] and r['Source'] and r['Evidence']
        assert '\ufffd' not in r['Form']
        assert unicodedata.normalize('NFC', r['Form']) == r['Form']
    return [[r[k] for k in ['ID', 'Language_ID', 'Form', 'Gloss', 'Source']] for r in records]

def render():
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows(rows())
    return buf.getvalue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    result = render()
    if args.install:
        OUTPUT.write_bytes(result.encode())
    else:
        assert OUTPUT.read_bytes() == result.encode()
    print('406 approved donor heads; deterministic preservation and provenance checks passed')
