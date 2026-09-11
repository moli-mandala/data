"""Reproduce the 33 curated donor heads for the approved Dameli review (2026-09-10).

Input is the checked, source-attributed editorial audit, not a live scrape.
Only reviewed lexical heads are installed; full dictionary coverage is excluded.
The parameter format is intentional: these are curated donor etyma, not new
Dameli attestations. Preserve source transcription, applying NFC only; raw
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
AUDIT = Path(__file__).with_name('20260910-dameli-donors-audit.json')
OUTPUT = ROOT / 'data/other/params/20260910-dameli-donors.csv'
FORMS = ROOT / 'data/other/forms/20260910-dameli-donors.csv'

def rows():
    records = json.loads(AUDIT.read_text())
    assert len(records) == 33
    assert len({r['ID'] for r in records}) == len(records)
    for r in records:
        assert r['Status'] == 'approved'
        assert r['Language_ID'] in {'Psht', 'H'}
        assert r['Form'] and r['Gloss'] and r['Source'] and r['Evidence']
        assert '\ufffd' not in r['Form']
        assert unicodedata.normalize('NFC', r['Form']) == r['Form']
    return [[r[k] for k in ['ID', 'Language_ID', 'Form', 'Gloss', 'Source']] for r in records]

def render():
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows(rows())
    return buf.getvalue()

def render_forms():
    # Matching self-attestations fold native spelling and grammar onto the curated head.
    buf = io.StringIO(newline='')
    writer = csv.writer(buf)
    for r in json.loads(AUDIT.read_text()):
        writer.writerow([r['Language_ID'], r['ID'], r['Form'], r['Gloss'],
                         r.get('Native', ''), '', '', r['Source'], '',
                         r.get('Source_Etymology', ''), r['ID'] + ':attestation',
                         '', '', '', r.get('Tags', '')])
    return buf.getvalue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    result = render()
    if args.install:
        OUTPUT.write_bytes(result.encode())
        FORMS.write_bytes(render_forms().encode())
    else:
        assert OUTPUT.read_bytes() == result.encode()
        assert FORMS.read_bytes() == render_forms().encode()
    print('33 approved donor heads; deterministic preservation and provenance checks passed')
