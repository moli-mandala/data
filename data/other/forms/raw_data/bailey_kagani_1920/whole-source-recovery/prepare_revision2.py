"""Reconcile failed r1 metadata into a separate deterministic r2 candidate."""
import csv,json,hashlib,shutil,re
from pathlib import Path
P=Path(__file__).resolve().parent;F=P/'whole-proposal-r1';O=P/'revision2-working'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    freeze=json.loads((F/'review-manifest.json').read_text())
    for name,digest in freeze['output_sha256'].items():assert sha(F/name)==digest,name
    review=json.loads((P/'revision2-preparation/interlinear-metadata-class-review.json').read_text())
    for name,digest in review['input_sha256'].items():assert sha(F/name)==digest,name
    root=json.loads((F/'root-output-review-20260927.json').read_text())
    assert root['status']=='failed'
    rows=list(csv.reader((F/'proposal.csv').open()));old=[r[:] for r in rows];keyed={r[10]:r for r in rows}
    applied={}
    for delta in review['changes']:
        r=keyed[delta['entry_key']]
        assert (r[2],r[3],r[14])==(delta['old_form'],delta['old_gloss'],delta['old_tags'])
        r[3]=delta['new_gloss'];r[14]=delta['new_tags'];applied[r[10]]=delta['basis']
    for r in rows:
        original=r[6]
        r[6]=re.sub(r'Some proposed answers are multiword; apply multiword-expression per emitted answer\.\s*','',r[6])
        r[6]=re.sub(r'Apply multiword-expression[^.]*\.\s*','',r[6])
        r[6]=r[6].replace('Printed source qualifier preserved in Tags. ','').strip()
        if r[6]!=original:applied.setdefault(r[10],[]).append('Removed procedural parser/tagging prose; retained source grammar, expansion and context evidence.')
    assert [r[10] for r in rows]==[r[10] for r in old]
    assert [r[2] for r in rows]==[r[2] for r in old]
    audit=[json.loads(s) for s in(F/'proposal-audit.jsonl').read_text().splitlines()]
    for unit in audit:
        changed=[k for k in unit.get('entry_keys',[]) if k in applied]
        if changed:
            unit['r2_metadata_reconciliation']=[{'entry_key':k,'exported_gloss':keyed[k][3],'exported_notes':keyed[k][6],'exported_tags':keyed[k][14],'basis':applied[k]} for k in changed]
    O.mkdir(exist_ok=True)
    with (O/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (O/'proposal-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    shutil.copyfile(F/'profile-proposal.txt',O/'profile-proposal.txt')
    changes=[{'entry_key':b[10],'fields':{str(i):{'old':x,'new':y} for i,(x,y) in enumerate(zip(a,b)) if x!=y},'basis':applied[b[10]]} for a,b in zip(old,rows) if a!=b]
    manifest={'status':'r2 working candidate; fresh independent audit required','rows':len(rows),'audit_units':len(audit),'all_r1_keys_forms_stable':True,'changes':changes,'input_sha256':{'r1/review-manifest.json':sha(F/'review-manifest.json'),'r1/root-output-review-20260927.json':sha(F/'root-output-review-20260927.json'),'revision2-preparation/interlinear-metadata-class-review.json':sha(P/'revision2-preparation/interlinear-metadata-class-review.json'),'prepare_revision2.py':sha(Path(__file__))},'output_sha256':{n:sha(O/n) for n in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']}}
    (O/'revision-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'audit_units':len(audit),'changed_rows':len(changes),'hashes':manifest['output_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
