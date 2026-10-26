"""Create a source-local acceptance derivative; never install canonical files.

Acceptance concerns each accounting decision, including explicit holds and
documentary exclusions. It does not promote held candidates to lexical rows.
All original fields and nested historical statuses are preserved unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'integration-proposal/whole-source-witness-audit.jsonl'

def sha(b): return hashlib.sha256(b).hexdigest()

def main(expected):
    raw = SOURCE.read_bytes()
    assert sha(raw) == expected, 'Assembly changed since root freeze'
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 6823 == len({r['audit_key'] for r in rows})
    assert all('accepted_for_installation' not in r for r in rows)
    accepted = [{**r, 'accepted_for_installation': True,
                 'acceptance_scope': 'Reviewed source accounting decision; held candidates remain held'} for r in rows]
    assert all({k:v for k,v in new.items() if k not in ('accepted_for_installation','acceptance_scope')} == old
               for old,new in zip(rows,accepted))
    output = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted).encode()
    target = HERE / 'accepted-audit.jsonl'
    if target.exists(): assert target.read_bytes() == output, 'Refuse replacing prior derivative'
    else: target.write_bytes(output)
    report = {'status':'pass','input_sha256':expected,'output_sha256':sha(output),
              'audit_units':len(rows),'all_original_fields_preserved':True,
              'accepted_decisions_not_held_lexemes':True,'canonical_changes':False}
    (HERE/'accepted-audit-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-input-sha256',required=True)
    main(parser.parse_args().expected_input_sha256)
