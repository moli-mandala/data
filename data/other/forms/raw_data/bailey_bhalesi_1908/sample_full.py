"""Reproducible full-source sample; read-only except the requested sample file."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent
FILES = ('full-reviewed.jsonl', 'proposal.csv', 'proposal-audit.jsonl', 'proposal-profile.txt')


def hashes():
    return {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    args = parser.parse_args()
    frozen = json.loads((P / 'full-stage-freeze-20260926.json').read_text())['hashes']
    assert hashes() == frozen, 'Frozen staging changed'
    excluded = set()
    # The earlier bounded review sampled glossary items; never reuse those units.
    for row in csv.DictReader((P / 'sample-review-20.tsv').open(), delimiter='\t'):
        item = int(row['item'])
        page = 73 if item <= 12 else 74
        column = 'left' if item <= 6 or 13 <= item <= 23 else 'right'
        excluded.add(f'bailey1908bhalesi:p{page}:{column}:item:{item}')
    for path in args.exclude:
        report = json.loads(path.read_text())
        excluded.update(report.get('selected_keys', []))
        excluded.update(u['source_unit_key'] for u in report.get('sample', []))
    audit = [json.loads(line) for line in (P / 'proposal-audit.jsonl').read_text().splitlines()]
    candidates = [u for u in audit if u['entry_keys'] and u['source_unit_key'] not in excluded]
    selected = random.Random(args.seed).sample(candidates, 20)
    rows = {r[10]: r for r in csv.reader((P / 'proposal.csv').open())}
    output = {'seed': args.seed, 'hashes': frozen, 'excluded_keys': sorted(excluded),
              'selected_keys': [u['source_unit_key'] for u in selected],
              'sample': [{**u, 'proposed_rows': [rows[k] for k in u['entry_keys']]} for u in selected]}
    assert hashes() == frozen
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
