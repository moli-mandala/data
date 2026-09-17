"""Reproduce five approved batch-009 family and donor/component heads.

This emits only the parameter supplement. Attestations and Hindi comparanda
are existing records retained in the adjacent audit; no build is invoked.
"""
import csv
import io
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).with_name('20260911-mewari-batch009-heads-audit.json')
OUTPUT = ROOT / 'data/other/params/20260911-mewari-batch009-heads.csv'


def render():
    records = json.loads(AUDIT.read_text())
    assert len(records) == len({r['Entry_Key'] for r in records}) == 5
    buf = io.StringIO(newline='')
    for row in records:
        assert row['Language_ID'] in {'Rj', 'Indo-Aryan'}
        assert row['Etymology'] and (row['Attestations'] or row['Role']=='donor')
        assert unicodedata.normalize('NFC', row['Form']) == row['Form']
        assert '\ufffd' not in row['Form']
        csv.writer(buf).writerow([row[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source')])
    return buf.getvalue().encode()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if args.install:
        OUTPUT.write_bytes(render())
    else:
        assert OUTPUT.read_bytes() == render()
    print('Five family/donor/component heads reproduced; no build run.')
