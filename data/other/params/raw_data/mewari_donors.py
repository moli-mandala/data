"""Emit the selected Mewari donor dictionary heads from the checked source audit.

Five-column curated parameters follow the existing Pashto donor supplements.
No fabricated regional attestations, new language IDs, or live scraping. Source
spellings, native scripts, sense exclusions and explicit transcription choices
are retained in the audit. --install writes only this source's parameter CSV.
"""
import argparse
import csv
import io
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).with_name('20260911-mewari-donors-audit.json')
OUTPUT = ROOT / 'data/other/params/20260911-mewari-donors.csv'

def render():
    records = json.loads(AUDIT.read_text())
    assert len(records) == 22
    assert len({r['Entry_Key'] for r in records}) == len(records)
    selected = [r for r in records if r['Status'] == 'install']
    assert len(selected) == 13
    for r in records:
        assert r['Language_ID'] in {'H', 'Indo-Aryan'}
        assert all(r[k] for k in ('Original', 'Gloss', 'Source', 'Evidence', 'URL', 'Transcription'))
        assert unicodedata.normalize('NFC', r['Form']) == r['Form']
        assert '\ufffd' not in r['Form']
        if r['Status'] == 'reuse':
            assert r['Persistent_ID'].startswith('f_')
    buf = io.StringIO(newline='')
    csv.writer(buf).writerows([[r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source')] for r in selected])
    return buf.getvalue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    out = render().encode()
    if args.install:
        OUTPUT.write_bytes(out)
    else:
        assert OUTPUT.read_bytes() == out
    print('22 dictionary heads audited: 13 emitted, 9 reused')
