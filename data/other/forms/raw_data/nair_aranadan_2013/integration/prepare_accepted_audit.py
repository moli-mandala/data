"""Derive canonical audit only after explicit root source acceptance; never install."""
from pathlib import Path
import csv,hashlib,json
H=Path(__file__).resolve().parent
P=H.parent
APPROVAL=P/'root-review/source-acceptance.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    approval=json.loads(APPROVAL.read_text())
    assert approval['approved'] is True and approval['result']=='pass'
    required={'integration/review-candidate.csv','integration/review-audit.jsonl','root-review/approved-reconciliation.json'}
    assert required<=set(approval['files']) and approval['review_reports']
    assert set(approval['review_reports'])<=set(approval['files'])
    for name,value in approval['files'].items():
        path=(P/name).resolve();assert path.is_relative_to(P.resolve())
        assert sha(path)==value,(name,'changed accepted evidence')
    rows=list(csv.reader((H/'review-candidate.csv').open()))
    original=[json.loads(line) for line in (H/'review-audit.jsonl').read_text().splitlines()]
    assert len(rows)==102 and len(original)==97
    keys={r[10] for r in rows};assert len(keys)==102
    result=[]
    for old in original:
        record={k:v for k,v in old.items() if k not in ('accepted_for_installation','pending','disposition','review_candidate_keys')}
        entry_keys=old['review_candidate_keys']
        record.update(entry_keys=entry_keys,accepted_for_installation=True,
            disposition='emitted' if entry_keys else 'documentary-excluded',
            acceptance=dict(path=str(APPROVAL.relative_to(P)),sha256=sha(APPROVAL),result='pass'),
            historical_review=dict(path='integration/review-audit.jsonl',sha256=sha(H/'review-audit.jsonl'),
                prior_disposition=old['disposition']))
        result.append(record)
    mapped=[k for r in result for k in r['entry_keys']]
    assert set(mapped)==keys and len(mapped)==102
    assert len({r['source_unit_key'] for r in result})==97
    assert sum(r['physical_witness_slots'] for r in result)==105
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result)
    output=H/'accepted-audit.jsonl'
    if output.exists():assert output.read_text()==content,'Accepted derivative drift; require explicit new version.'
    else:output.write_text(content)
    report=dict(status='source-stage audit accepted; compiled gates deferred',rows=102,audit_units=97,physical_witness_slots=105,
        target_rows=81,comparison_rows=21,documentary_units=sum(not r['entry_keys'] for r in result),
        source_approval_sha256=sha(APPROVAL),accepted_csv_sha256=sha(H/'review-candidate.csv'),accepted_audit_sha256=sha(output),
        original_review_audit_sha256=sha(H/'review-audit.jsonl'),all_emitted_keys_closed=True,
        deferred=['Full CLDF/database build','Full suite','Compiled graph and durable identity validation','Browser QA','Publication'])
    (H/'accepted-audit-status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
