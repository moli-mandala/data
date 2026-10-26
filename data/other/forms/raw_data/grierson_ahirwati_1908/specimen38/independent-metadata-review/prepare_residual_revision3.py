from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;OLD=P/'residual-revision2';NEW=P/'residual-revision3';NEW.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD/'independent-full-residual-metadata-review.json')=='e4817e4e61b5fcf12ba3c538331eb8ebf635e16e0499358d47348883431f962b'
rows=[json.loads(s) for s in (OLD/'all-132-keyed-metadata-decisions.jsonl').read_text().splitlines()]
key='lsi9.2:p238:aligned-line:10:word:3'
oldrow=json.loads(json.dumps(next(r for r in rows if r['source_unit_key']==key)))
assert oldrow['exact_form_guard']=='kahā' and oldrow['exact_gloss_guard']=='it-was-said' and 'pass' in oldrow['proposed_tags']
r=next(r for r in rows if r['source_unit_key']==key)
r['proposed_tags'].remove('pass')
r['additional_scope_holds'].append({'type':'morphological-voice-scope','detail':'Printed it-was-said is retained as lexical English analysis; preceding jamāī-nai agent context admits active/impersonal interpretations, so morphological passive is not established.'})
r['source_scoped_basis'].append('Root independently withholds morphological pass for this exact source occurrence; English passive translation alone is insufficient.')
for name,rr in [('all-132-keyed-metadata-decisions.jsonl',rows),('keyed-tag-change-proposals.jsonl',[r for r in rows if r['old_tags']!=r['proposed_tags']])]:
 (NEW/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
report=json.loads((OLD/'independent-full-residual-metadata-review.json').read_text())
report['status']='Complete132 residual review, root-approved scoped voice exception applied to reviewer proposal only; frozen source inputs unchanged'
report['previous_review_sha256']=sha(OLD/'independent-full-residual-metadata-review.json')
report['revision3_single_case']={'key':key,'old':oldrow,'new':r,'guard':'Only this recommendation changes from residual2; all132 exact literal/gloss/old-tag guards unchanged.'}
report['outputs']={f.name:sha(f) for f in NEW.iterdir() if f.is_file()}
report['review_script_sha256']=sha(Path(__file__))
(NEW/'independent-full-residual-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(sha(NEW/'all-132-keyed-metadata-decisions.jsonl'));print(sha(NEW/'independent-full-residual-metadata-review.json'))
