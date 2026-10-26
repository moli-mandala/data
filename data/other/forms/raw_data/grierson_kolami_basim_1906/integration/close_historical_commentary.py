"""Guard three root-approved current-commentary repairs; preserve old revisions."""
import copy,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,rr):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def main():
    review=PACKAGE/'grammar-native-review/sharp-revision2-history-review.json'
    assert sha(review)=='8954136b44ad6e22a158cdd30dfcd2fca2c14757d272f936bd6077601b28758e'
    findings=json.loads(review.read_text())['keyed_display_findings'];assert len(findings)==3
    old=PACKAGE/'grammar-review/sharp-reviewed-revision2'
    out=PACKAGE/'grammar-review/sharp-reviewed-revision3';out.mkdir(exist_ok=True)
    children=rows(old/'reviewed-child-witnesses.jsonl');graphs=rows(old/'grammar-graph-proposal.jsonl')
    cb={r['source_unit_key']:r for r in children};gb={r['witness_key']:r for r in graphs}
    for d in findings:
        r=cb[d['key']] if d['field']=='qualification' else gb[d['key']]
        assert r[d['field']]==d['old'],d
        r.setdefault('historical_display_commentary',{})[d['field']]=d['old']
        r[d['field']]=d['new']
    write(out/'reviewed-child-witnesses.jsonl',children);write(out/'grammar-graph-proposal.jsonl',graphs)
    shutil.copyfile(old/'grammar-tag-proposal.jsonl',out/'grammar-tag-proposal.jsonl')
    before=HERE/'revision2';after=HERE/'revision3';after.mkdir(exist_ok=True)
    with (before/'combined-review-candidate.csv').open() as f:original=list(csv.reader(f))
    rr=copy.deepcopy(original);by={r[10]:r for r in rr};changes=[]
    for d in findings:
        r=by[d['key']];assert r[6].count(d['old'])==1,(d['key'],r[6])
        previous=r[6];r[6]=r[6].replace(d['old'],d['new'])
        changes.append({'source_unit_key':d['key'],'field':'Notes','old':previous,'new':r[6],'exact_review_decision':d})
    assert sum(a!=b for a,b in zip(original,rr))==3
    assert all(a[:6]+a[7:]==b[:6]+b[7:] for a,b in zip(original,rr))
    audit=rows(before/'combined-review-audit.jsonl')
    for a in audit:
        k=a['source_unit_key']
        if k in {d['key'] for d in findings}:
            a['historical_display_source_record']=copy.deepcopy(a['source_record'])
            a['source_record']=cb[k]
            a['actual_row']=by[k]
            a['root_approved_current_commentary_review']={'path':str(review),'sha256':sha(review),'approval':'Root task message 2026-09-27 explicitly accepts these three exact field repairs.'}
    for n in ['grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml','specimen-metadata-proposal.jsonl']:
        shutil.copyfile(before/n,after/n)
    with (after/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(rr)
    write(after/'combined-review-audit.jsonl',audit)
    report={'status':'Three exact root-approved current commentary repairs; historical claims remain in prior revisions and explicit history',
            'inputs':{str(p):sha(p) for p in [review,before/'combined-review-candidate.csv',old/'reviewed-child-witnesses.jsonl',old/'grammar-graph-proposal.jsonl']},
            'deltas':changes,'rows':827,'changed_notes':3,'other_fields_changed':0,'canonical_changes':False}
    (after/'commentary-continuity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    status=json.loads((before/'status.json').read_text())
    status['status']='Revision3 approved tag/commentary repairs; residual metadata review and final original-output audit pending'
    for name in ['reviewed-child-witnesses.jsonl','grammar-tag-proposal.jsonl','grammar-graph-proposal.jsonl']:
        status['input_pins'][str((out/name).relative_to(PACKAGE))]=sha(out/name)
    status['input_pins']['grammar-native-review/sharp-revision2-history-review.json']=sha(review)
    status['outputs']={n:sha(after/n) for n in ['combined-review-candidate.csv','combined-review-audit.jsonl','specimen-metadata-proposal.jsonl','grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']}
    status['pending'].insert(0,'Asur complete remaining case/object/residual metadata review and root adjudication')
    (after/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':827,'Notes_changes':3,'csv_sha256':sha(after/'combined-review-candidate.csv')}))
if __name__=='__main__':main()
