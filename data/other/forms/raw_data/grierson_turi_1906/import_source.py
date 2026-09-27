"""Reproduce the complete LSI IV Turi chapter, printed pp. 128–134."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = '20260925-grierson-turi-sites'
SOURCE = 'grierson1906lsi4'
_spec = importlib.util.spec_from_file_location('turi_full_review', ROOT / 'prepare_full_review.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DIALECTS = _module.SITES


def build():
    rows, audit = _module.build()
    assert len(rows) == 291 and len(audit) == 352
    assert len({r[10] for r in rows}) == len(rows)
    assert all(len(r) == 15 and r[0] == 'Turi' for r in rows)
    return rows, audit


def write(install=False):
    rows, audit = build()
    path = ROOT.parents[1] / (STEM + '.csv') if install else ROOT / 'full-review.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(rows)
    if install:
        (ROOT / 'full-installed-audit.jsonl').write_text(''.join(
            json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    print(f'{len(rows)} Turi forms; {len(audit)} source units; no database built')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    write(parser.parse_args().install)
