"""Guard the complete residual metadata review into a new revision after root approval.

Run only with an exact root artifact. Earlier candidates and failed review history
remain unchanged; every protected lexical/graph field is compared before output.
"""
import argparse,copy,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,rr):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def main(approval):
    assert sha(approval)=='578e5cc3f22512afb1ba42660fda2f669b25f2b0650474cc9107bc74165f1768'
    a=json.loads(approval.read_text())
    assert a['status']=='accepted_source_local_metadata_revision'
    source=HERE/'revision3';csvpath=source/'combined-review-candidate.csv'
    review_directory=(PACKAGE/a['residual_review_directory']).resolve()
    assert review_directory.is_relative_to((PACKAGE/'specimen-metadata-independent-review').resolve())
    report=review_directory/'residual-complete-scope-review.json'
    decisions=review_directory/'residual-keyed-decisions.jsonl'
    full=review_directory/'all-560-final-metadata-proposal.jsonl'
    assert a['candidate_sha256']==sha(csvpath)=='c81089759ddeb215f00208662413de1e855ea8d25120a12becaf9ff664e4b13f'
    assert a['residual_report_sha256']==sha(report)
    assert a['residual_decisions_sha256']==sha(decisions)
    assert a['final_metadata_sha256']==sha(full)
    dd=rows(decisions);metadata=rows(full)
    assert len(dd)==a['residual_decision_count'] and len({d['source_unit_key'] for d in dd})==len(dd)
    assert set(a['accepted_source_unit_keys'])=={d['source_unit_key'] for d in dd}
    assert len(metadata)==len({m['source_unit_key'] for m in metadata})==560
    with csvpath.open() as f:old=list(csv.reader(f))
    rr=copy.deepcopy(old);by={r[10]:r for r in rr};mb={m['source_unit_key']:m for m in metadata}
    for d in dd:
        key=d['source_unit_key'];r=by[key]
        assert r[2]==d['exact_form_guard'] and r[3]==d['exact_raw_gloss_guard']
        assert r[14].split()==d['old'],(key,r[14],d['old'])
        new=d['new'][:]
        for tag in r[14].split():
            if tag=='uncertain' or tag.startswith('dialect:'):
                if tag not in new:new.append(tag)
        r[14]=' '.join(new)
    # All560 final owner-independent decisions are checked, not merely changed79.
    for key,m in mb.items():
        assert by[key][2]==m['exact_form_guard'] and by[key][3]==m['exact_raw_gloss_guard']
        assert set(by[key][14].split())==set(m['proposed_tags'])
    assert a['notes_policy']=='exact_source_notes_overlay'
    nd=a['notes_decisions'];assert len({d['source_unit_key'] for d in nd})==len(nd)
    for d in nd:
        r=by[d['source_unit_key']]
        assert r[2]==d['exact_form_guard'] and r[3]==d['exact_raw_gloss_guard']
        assert r[6]==d['old_notes']
        r[6]=d['new_notes']
    assert all(x[:6]+x[7:14]==y[:6]+y[7:14] for x,y in zip(old,rr))
    audit=rows(source/'combined-review-audit.jsonl')
    for unit in audit:
        key=unit['source_unit_key']
        if key in mb and unit.get('kind')=='atomic':
            unit['accepted_residual_metadata']={'approval_path':str(approval),'approval_sha256':sha(approval),'decision':mb[key]}
            unit['typed_metadata_uncertainties']=mb[key]['typed_metadata_uncertainties']
            unit['scholarly_source_claims']=mb[key]['scholarly_source_claims']
            unit['typed_uncertainties']+=copy.deepcopy(mb[key]['typed_metadata_uncertainties'])
            unit['actual_row']=by[key]
    out=HERE/'revision4';out.mkdir(exist_ok=True)
    for n in ['grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']:
        shutil.copyfile(source/n,out/n)
    with (out/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(rr)
    write(out/'combined-review-audit.jsonl',audit);write(out/'specimen-metadata-proposal.jsonl',metadata)
    changed=[{'source_unit_key':x[10],'old_tags':x[14],'new_tags':y[14]} for x,y in zip(old,rr) if x[14]!=y[14]]
    continuity={'status':'Complete residual source-scoped metadata accepted; final actual output review still pending',
        'inputs':{str(p):sha(p) for p in [approval,report,decisions,full,csvpath]},'counts':{'rows':827,'reviewed_atomic_rows':560,'keyed_decisions':len(dd),'changed_tags':len(changed),'changed_notes':sum(x[6]!=y[6] for x,y in zip(old,rr)),'typed_scope_holds':sum(bool(m['typed_metadata_uncertainties']) for m in metadata)},
        'all_other13fields_keys_order_literals_glosses_graphs_citations_unchanged':True,'deltas':changed,'exact_notes_decisions':nd,'canonical_changes':False}
    (out/'residual-continuity.json').write_text(json.dumps(continuity,ensure_ascii=False,indent=2)+'\n')
    status=json.loads((source/'status.json').read_text())
    status['status']='Complete source-local reviewed candidate; final original-output audit and installation approval pending'
    status['input_pins']['residual_root_approval']=sha(approval)
    status['outputs']={n:sha(out/n) for n in ['combined-review-candidate.csv','combined-review-audit.jsonl','specimen-metadata-proposal.jsonl','grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']}
    status['pending']=['Final actual-parser/profile/reference/registration checks','Fresh independent actual-output original audit','Guarded source-stage installation approval','Full compiled/browser gates deferred']
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':827,'changed_tags':len(changed),'csv_sha256':sha(out/'combined-review-candidate.csv')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True,type=Path);a=p.parse_args();main(a.approval)
