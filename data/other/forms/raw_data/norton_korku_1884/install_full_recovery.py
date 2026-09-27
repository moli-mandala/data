"""Install the independently approved whole-source proposal; never build a database."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    freeze = json.loads((P/'full-recovery-freeze.json').read_text())
    approval = json.loads((P/'independent-full-recovery-audit-20260926-pass1.json').read_text())
    assert approval['status'] == 'passed' and approval['material_errors'] == 0 and approval['sample_size'] == 20
    for name, digest in freeze['hashes'].items():
        assert sha(P/name) == digest == approval['hashes'][name], name
    evidence = json.loads((P/'full-recovery-evidence-manifest.json').read_text())
    for name, digest in evidence['hashes'].items():
        assert sha(P/name) == digest, name
    reconciled = json.loads((P/'full-forward-reconciled.json').read_text())
    for name, digest in reconciled['input_hashes'].items():
        assert sha(P/name) == digest, name
    rows = list(csv.reader((P/'full-recovery-proposal.csv').open()))
    audit = [json.loads(s) for s in (P/'full-recovery-proposal-audit.jsonl').read_text().splitlines()]
    rr = {r[10]:r for r in rows}
    assert len(rows) == len(rr) == 1076 and len(audit) == 958
    assert len(set(reconciled['reviewed_keys'])) == 477
    for fix in reconciled['corrections']:
        matching = [r for r in rows if (r[10] == fix['entry_key'] or r[10].startswith(fix['entry_key']+':')) and r[2] == fix['form']]
        assert len(matching) == 1, fix
    for rejected in reconciled['rejected_corrections']:
        assert any(r[2] == rejected['accepted_form'] and r[10].startswith(rejected['entry_key']) for r in rows)
    assert not any(a['status'] in {'held','excluded_sentence'} for a in audit)
    for a in audit:
        assert set(a['entry_keys']) <= rr.keys()
        assert bool(a['entry_keys']) == (a['status'] not in {'excluded_control','audit_only_metalinguistic'})
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows, audit = validate()
    if args.install:
        for src, name in [(DATA/'data/other/forms/20260925-cust-norton-korku.csv','canonical-before-full-recovery.csv'),(P/'audit.jsonl','audit-before-full-recovery.jsonl')]:
            if not (P/name).exists():
                shutil.copyfile(src, P/name)
        for name, target in [('full-recovery-proposal.csv',DATA/'data/other/forms/20260925-cust-norton-korku.csv'),('full-recovery-proposal-audit.jsonl',P/'audit.jsonl'),('full-recovery-profile.txt',DATA/'conversion/cust-norton-korku-1884.txt'),('full-recovery-source.yaml',DATA/'data/other/forms/20260925-cust-norton-korku.yaml')]:
            shutil.copyfile(P/name, target)
    print(f'{len(rows)} rows / {len(audit)} units validated; no database or full build')


if __name__ == '__main__':
    main()
