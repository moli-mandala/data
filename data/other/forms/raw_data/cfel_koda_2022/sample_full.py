"""Emit a reproducible independent raw/output sample; do not render or install."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--count', default=20, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    units = [json.loads(line) for line in (ROOT / 'full-proposal-audit.jsonl').read_text().splitlines()]
    api_path = ROOT.parent / 'cfel_koda_api_2026' / 'full-proposal-audit.jsonl'
    api = {x.get('paired_print_key', x['entry_key']): x for x in map(json.loads, api_path.read_text().splitlines())}
    selected = random.Random(args.seed).sample(units, args.count)
    for item in selected:
        item['publisher_attestation'] = api[item['entry_key']]
    paths = [ROOT / 'full-proposal.csv', ROOT / 'full-proposal-audit.jsonl', api_path, api_path.with_name('full-proposal.csv')]
    result = {'seed': args.seed, 'population': len(units), 'sample_size': len(selected),
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'sample': selected}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)
