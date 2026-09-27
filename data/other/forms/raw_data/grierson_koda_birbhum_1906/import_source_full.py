"""Reproduce the whole LSI IV Koda source package, without building a database."""
import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = '20260925-grierson-koda-birbhum'
spec = importlib.util.spec_from_file_location('koda_full_assembler', ROOT / 'prepare_full_preview.py')
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)


def build():
    return assembler.build()


def write(install=False):
    rows, audit = build()
    destination = ROOT / f'{STEM}.csv'
    with destination.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(rows)
    audit_path = ROOT / 'full-installed-audit.jsonl'
    with audit_path.open('w', encoding='utf-8') as handle:
        for unit in audit:
            handle.write(json.dumps(unit, ensure_ascii=False) + '\n')
    if install:
        shutil.copyfile(destination, ROOT.parents[1] / destination.name)
    print(f'{len(rows)} Koda forms from {len(audit)} physical source units; no database built')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    write(parser.parse_args().install)
