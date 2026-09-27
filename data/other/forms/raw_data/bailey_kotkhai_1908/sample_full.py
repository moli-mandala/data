"""Select a reproducible, stratified 20-unit final-output audit sample."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent


def select(seed):
    rows = {r[10]: r for r in csv.reader((P / 'proposal.csv').open())}
    units = [json.loads(line) for line in (P / 'proposal-audit.jsonl').read_text().splitlines()]
    strata = {k: [] for k in ['p23-paradigms', 'p23-adverbs', 'p24-auxiliary', 'p24-main']}
    for unit in units:
        if not unit['entry_keys']:
            continue
        if unit['printed_page'] == 23:
            stratum = 'p23-adverbs' if unit['section'] == 'adverb' else 'p23-paradigms'
        else:
            stratum = 'p24-auxiliary' if unit['section'] == 'auxiliary' else 'p24-main'
        strata[stratum].append(unit)
    rng = random.Random(seed)
    sample = []
    for stratum, pool in strata.items():
        for unit in rng.sample(pool, 5):
            sample.append({**unit, 'audit_stratum': stratum,
                           'actual_csv_rows': [rows[k] for k in unit['entry_keys']]})
    return {'status': 'selected_pending_independent_review', 'seed': seed, 'sample_size': 20,
            'hashes': {f: hashlib.sha256((P / f).read_bytes()).hexdigest() for f in
                       ['proposal.csv', 'proposal-audit.jsonl', 'proposal-profile.txt']}, 'rows': sample}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True, help='Filename inside this source package')
    args = parser.parse_args()
    assert Path(args.output).name == args.output
    (P / args.output).write_text(json.dumps(select(args.seed), ensure_ascii=False, indent=2) + '\n')
