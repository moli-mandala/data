"""Freeze a disjoint seeded actual-output sample; never modify lexical inputs."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--exclude', type=Path, action='append', default=[])
    args = ap.parse_args()
    pinned = ['proposal.csv','proposal-audit.jsonl','proposal-profile.txt','proposal.yaml',
              'about-attestations.json','lexical-snapshot.jsonl','lexical-context.json',
              'prepare_source.py','proposed-reference.bib','proposed-dialect.csv','media-url-provenance.json']
    hashes = {f: hashlib.sha256((P/f).read_bytes()).hexdigest() for f in pinned}
    bykey = {r[10]:r for r in csv.reader((P/'proposal.csv').open())}
    audit = [json.loads(l) for l in (P/'proposal-audit.jsonl').read_text().splitlines()]
    excluded = set()
    for path in args.exclude:
        old = json.loads(path.read_text())
        assert isinstance(old['sample'], list)
        excluded.update(a['source_unit_key'] for a in old['sample'])
    candidates = [a for a in audit if a['entry_keys'] and a['source_unit_key'] not in excluded]
    chosen = random.Random(args.seed).sample(candidates, 20)
    def enrich(a):
        return dict(a, actual_output_rows=[bykey[k] for k in a['entry_keys']])
    controls = [a for a in audit if a['section'] == 'about' or a.get('borrowing_claims') or
                a.get('source_pos') in (['adp'], ['sent'], ['adv.p']) or len(a['entry_keys']) > 1]
    out = {'status':'awaiting independent review', 'seed':args.seed, 'pinned_sha256':hashes,
           'excluded_source_unit_keys':sorted(excluded), 'sample':[enrich(a) for a in chosen],
           'targeted_control_keys':[a['source_unit_key'] for a in controls],
           'original_snapshot':'snapshot/kera-mundari.db', 'original_about':'snapshot/about.html',
           'instructions':'Compare with raw SQLite rows and original about HTML, not only the transformed JSON export. Check labels, gloss qualifiers, literal spelling and variant scope. Media paths require no playback or download.'}
    path = P / f'independent-selection-{args.seed}.json'
    assert not path.exists(), 'Preserve prior selections; use a new seed.'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(path)

if __name__ == '__main__': main()
