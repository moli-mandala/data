"""Regenerate and install the independently accepted whole Korava source stage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from prepare_full import generate
P = Path(__file__).resolve().parent
DATA = P.parents[4]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--check-scan', action='store_true')
    args = parser.parse_args()
    freeze = json.loads((P / 'full-stage-freeze-20260926.json').read_text())
    for name, expected in freeze['hashes'].items():
        assert hashlib.sha256((P / name).read_bytes()).hexdigest() == expected, name
    rows, audit = generate()
    assert rows == list(csv.reader((P / 'proposal.csv').open()))
    assert audit == [json.loads(x) for x in (P / 'proposal-audit.jsonl').read_text().splitlines()]
    if args.check_scan:
        original = DATA.parent / 'tmp/pdfs/LSI-V4.djvu'
        assert hashlib.sha256(original.read_bytes()).hexdigest() == '33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1'
    if args.install:
        approval = json.loads((P / 'root-final-installation-review-20260926.json').read_text())
        assert approval['status'] == 'approved_for_source_stage_installation'
        assert approval['hashes']['proposal.csv'] == freeze['hashes']['proposal.csv']
        (P.parent.parent / '20260925-grierson-korvi.csv').write_bytes((P / 'proposal.csv').read_bytes())
        (P / 'audit.jsonl').write_bytes((P / 'proposal-audit.jsonl').read_bytes())
        (DATA / 'conversion/grierson-korvi-1906.txt').write_bytes((P / 'proposal-profile.txt').read_bytes())
    print(f'{len(rows)} rows; {len(audit)} source units; ' + ('installed' if args.install else 'verified'))

if __name__ == '__main__':
    main()
