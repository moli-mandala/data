from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
load=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
allowed={'tags','notes','typed_uncertainty','source_analysis','gloss'}
deltas=[];hashes={}
for name,count in [('whole-expression-proposal.jsonl',117),('c5-column-proposal.jsonl',192),('continuous-semantic-proposal.jsonl',790)]:
 old=load(H/'revision1'/name);new=load(H/name)
 assert len(old)==len(new)==count
 assert [x['source_unit_key'] for x in old]==[x['source_unit_key'] for x in new]
 for a,b in zip(old,new):
  fields=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)]
  assert set(fields)<=allowed,(name,b['source_unit_key'],fields)
  if 'gloss' in fields:
   assert b['source_unit_key']=='behera2022korwakodaku:C.5:column:19.2:4'
   assert a['gloss']=='entered (drawn out for emphasis)' and b['gloss']=='entered'
   assert b['source_analysis']=='drawn out for emphasis' and a['raw_gloss']==b['raw_gloss']
  if fields:deltas.append(dict(file=name,key=b['source_unit_key'],fields=sorted(fields)))
 hashes[name]=sha(H/name)
whole={r['source_unit_key']:r for r in load(H/'whole-expression-proposal.jsonl')}
cols={r['source_unit_key']:r for r in load(H/'c5-column-proposal.jsonl')}
cont=load(H/'continuous-semantic-proposal.jsonl')
assert whole['behera2022korwakodaku:C.5:expression:1']['tags']==['multiword-expression']
for i in (3,4):
 r=cols[f'behera2022korwakodaku:C.5:column:1:{i}'];assert r['typed_uncertainty']==['source-word-alignment'] and r['tags']==['uncertain']
for n,i in [('18.1',8),('19.1',9)]:
 r=cols[f'behera2022korwakodaku:C.5:column:{n}:{i}'];assert r['tags']==['emph'] and not r['gloss'] and r['source_analysis']=='(emph)'
assert 'source-notation' in whole['behera2022korwakodaku:C.4:expression:6']['typed_uncertainty']
dot=[r for r in cont if 'sa.b' in r['form']];assert len(dot)==1 and 'source-notation' in dot[0]['typed_uncertainty']
assert all('uncertain' in r['tags'] for r in cont if r['typed_uncertainty'])
report=dict(status='passed',scope='Independent-review metadata reconciliation, exact field continuity and complete uncertainty-class checks',rows=1099,hashes=hashes,unchanged='All forms, raw glosses, physical source locators, ordered keys, language routing and whole-expression grouping are identical to revision1.',lexical_gloss_changes=1,performance_note_retained=True,changed_rows=len(deltas),changes=deltas,independent_report_sha256=sha(H.parent/'interlinear-independent-review/independent-review-pass1.json'),limitations=['Not independent acceptance of 790 continuous semantic children.','No canonical installation or full database validation.'])
(H/'review-repair-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='changes'},indent=2))
