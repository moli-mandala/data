"""Apply only hash-pinned root-approved keyed metadata to a NEW revision."""
import argparse,copy,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,rr):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def main(approval):
    assert sha(HERE/'combined-review-candidate.csv')=='d70237705abcb7bcdb59a32f1898ebb0007ad9d83cbedcdab85d2473f6fa0509'
    assert sha(HERE/'specimen-metadata-proposal.jsonl')=='61cba0010f2ea47878a4278b4261586eaee363c01e66288f1fc694e4620f78ca'
    report=PACKAGE/'specimen-metadata-independent-review/independent-full-specimen-metadata-review.json'
    recpath=PACKAGE/'specimen-metadata-independent-review/keyed-tag-recommendations.jsonl'
    approved=json.loads(approval.read_text())
    assert approved['status']=='accepted_source_local_metadata_revision'
    assert approved['review_report_sha256']==sha(report)
    assert approved['recommendations_sha256']==sha(recpath)
    recs=rows(recpath);assert len(recs)==138 and len({r['source_unit_key'] for r in recs})==138
    assert set(approved['accepted_source_unit_keys'])=={r['source_unit_key'] for r in recs}
    with (HERE/'combined-review-candidate.csv').open() as f:original=list(csv.reader(f))
    rr=copy.deepcopy(original);by={r[10]:r for r in rr}
    m=rows(HERE/'specimen-metadata-proposal.jsonl');mb={r['source_unit_key']:r for r in m}
    deltas=[]
    for rec in recs:
        k=rec['source_unit_key'];assert rec['field']=='proposed_tags'
        assert mb[k]['proposed_tags']==rec['old']
        assert mb[k]['exact_form_guard']==rec['exact_form_guard']==by[k][2]
        assert mb[k]['exact_raw_gloss_guard']==rec['exact_raw_gloss_guard']==by[k][3]
        before=by[k][14];new=rec['new'][:]
        # Every pre-existing glyph uncertainty and source-qualified dialect survives.
        for t in before.split():
            if t=='uncertain' or t.startswith('dialect:'):
                if t not in new:new.append(t)
        by[k][14]=' '.join(new)
        mb[k]['historical_metadata_proposal']=copy.deepcopy(mb[k])
        mb[k]['proposed_tags']=rec['new']
        mb[k]['root_accepted_metadata_review']={'approval_sha256':sha(approval),'exact_recommendation':rec}
        mb[k]['review_status']='Source-local metadata recommendation accepted by root; final integrated output audit pending'
        deltas.append({'source_unit_key':k,'old_tags':before,'new_tags':by[k][14],'source_evidence':rec})
    assert sum(a!=b for a,b in zip(original,rr))==138
    assert all(a[:14]==b[:14] for a,b in zip(original,rr))
    audit=rows(HERE/'combined-review-audit.jsonl')
    for a in audit:
        if a['source_unit_key'] in by and 'actual_row' in a:a['actual_row']=by[a['source_unit_key']]
        if a['source_unit_key'] in {r['source_unit_key'] for r in recs}:
            a['accepted_source_local_metadata_approval']={'path':str(approval),'sha256':sha(approval)}
    out=HERE/'revision2';out.mkdir(exist_ok=True)
    for n in ['grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']:
        shutil.copyfile(HERE/n,out/n)
    with (out/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(rr)
    write(out/'combined-review-audit.jsonl',audit);write(out/'specimen-metadata-proposal.jsonl',m)
    continuity={'status':'Exact root-approved metadata-only derivative; old integration review remains unchanged',
        'input_pins':{str(p):sha(p) for p in [approval,report,recpath,HERE/'combined-review-candidate.csv',HERE/'combined-review-audit.jsonl',HERE/'specimen-metadata-proposal.jsonl']},
        'counts':{'rows':827,'tag_fields_changed':138,'other_fields_changed':0},'deltas':deltas,
        'all_keys_order_literals_glosses_graphs_citations_unchanged':True,'profile_and_settings_byte_identical':True,'canonical_changes':False}
    (out/'metadata-continuity.json').write_text(json.dumps(continuity,ensure_ascii=False,indent=2)+'\n')
    status=json.loads((HERE/'status.json').read_text());status['status']='Reviewed metadata revision2; final original-output audit pending'
    status['input_pins']['root_metadata_approval']=sha(approval)
    status['outputs']={n:sha(out/n) for n in ['combined-review-candidate.csv','combined-review-audit.jsonl','specimen-metadata-proposal.jsonl','grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']}
    status['pending']=['Final actual-parser/profile revalidation','Fresh independent actual-output original audit','Guarded installer/source-stage approval','Compiled/browser gates deferred']
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':827,'changed_tag_fields':138,'csv_sha256':sha(out/'combined-review-candidate.csv')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True,type=Path);a=p.parse_args();main(a.approval)
