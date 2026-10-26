"""Derive accepted audit only from root's exact final original-output acceptance."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
FINAL=PACKAGE/'integration/revision6'
ACCEPTANCE=PACKAGE/'root-independent-review/scholarly-notes-r6-acceptance.json'
HISTORICAL_ACCEPTANCE=PACKAGE/'root-independent-review/fresh-final-output-original-audit.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(ACCEPTANCE)=='6c7b7ee4ee605a2800080bf37ca801f967c01c4f5a6fade8d2134e894d722547'
    assert sha(HISTORICAL_ACCEPTANCE)=='2bc4a5c8be3ec9d127d04211ac6729f5eb2451bcba132bb7c61c46b56c737864'
    accepted=json.loads(ACCEPTANCE.read_text())
    assert accepted['status']=='pass' and accepted['source_stage_candidate_accepted'] is True
    assert accepted['installation_approved'] is False
    assert accepted['historical_original_output_acceptance_sha256']==sha(HISTORICAL_ACCEPTANCE)
    assert accepted['continuity_sha256']==sha(FINAL/'scholarly-notes-continuity.json')
    assert accepted['decisions']==json.loads((FINAL/'scholarly-notes-continuity.json').read_text())['decisions']
    for name,field in [('combined-review-candidate.csv','candidate_sha256'),('combined-review-audit.jsonl','audit_sha256'),('review-freeze.json','review_freeze_sha256'),('fresh-output-selection-20260927.json','selection_sha256')]:
        assert sha(FINAL/name)==accepted[field]
    freeze=json.loads((FINAL/'review-freeze.json').read_text())
    for name,pin in freeze['files'].items():assert sha(PACKAGE/name)==pin,name
    original=[json.loads(s) for s in (FINAL/'combined-review-audit.jsonl').read_text().splitlines()]
    with (FINAL/'combined-review-candidate.csv').open() as f:keys={r[10] for r in csv.reader(f)}
    assert len(keys)==827 and len(original)==1570
    reused={k for a in original for k in a['entry_keys']}-keys;assert len(reused)==3
    result=[]
    for source in original:
        assert source['accepted_for_installation'] is False
        a=dict(source)
        a['historical_review_disposition']=source['disposition']
        a['accepted_for_installation']=True
        if not a['entry_keys']:disposition='accepted_documentary_or_hypothetical_control_not_exported'
        elif set(a['entry_keys'])<=reused:disposition='accepted_same_occurrence_existing_key_reuse_not_new_emission'
        elif a['kind']=='physical_atom_accounting':disposition='accepted_physical_witness_mapping_to_reviewed_atomic_emission'
        else:disposition='accepted_source_stage_emission'
        a['accepted_disposition']=disposition
        a['source_stage_acceptance']={'path':str(ACCEPTANCE.relative_to(PACKAGE)),'sha256':sha(ACCEPTANCE),
            'historical_original_output_acceptance':{'path':str(HISTORICAL_ACCEPTANCE.relative_to(PACKAGE)),'sha256':sha(HISTORICAL_ACCEPTANCE)},
            'scope':'Source-stage original-output acceptance plus exact two scholarly Notes repairs; full compiled/browser gates deferred'}
        a['source_record_provenance']='Source record retains immutable original and reviewed derivative history; nested provisional flags are historical, not current acceptance state.'
        assert all(a[k]==v for k,v in source.items() if k!='accepted_for_installation')
        result.append(a)
    assert {k for a in result for k in a['entry_keys']}==keys|reused
    (HERE/'accepted-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in result))
    report={'status':'Exact root-accepted audit derivative; controls/reuse explicitly retained without promoting unexported forms',
       'input_audit_sha256':sha(FINAL/'combined-review-audit.jsonl'),'acceptance_sha256':sha(ACCEPTANCE),
       'output_audit_sha256':sha(HERE/'accepted-audit.jsonl'),'audit_records':1570,'emitted_keys':827,'existing_reused_keys':sorted(reused),
       'all_raw_source_records_entry_keys_dispositions_and_current_metadata_preserved':True,'unexported_controls_remain_without_entry_keys':True,
       'installation_approved':False,'canonical_changes':False}
    (HERE/'accepted-audit-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
