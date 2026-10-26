"""Resolve printed intra-source references with explicit sense scope; no fuzzy matching."""
import json,re,copy,collections
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[json.loads(l) for l in (P/'witness-proposed.jsonl').open()]
pdf=[r for r in rows if r['representation']=='published PDF'];head={}
for r in pdf:
 if r['source_record']['kind']=='head' and not r['source_record'].get('cross_reference'):
  head.setdefault(r['source_record']['form'],[]).append(r)
decisions=[]
for r in rows:
 sr=r['source_record'];before=copy.deepcopy(r['candidates']);key=r['source_unit_key']
 if sr.get('cross_reference'):
  raw=sr['cross_reference'];target=raw.removeprefix('=').removeprefix('see ').removeprefix('unstressed form of ').strip()
  targets=head.get(target,[]);assert len(targets)==1,(key,raw,targets)
  targets=targets[0]['candidates']
  reason='Printed exact cross-reference to a unique main head; inherit all defined senses without inventing a sense.'
  if raw=='see vazile':
   assert sr['parent_head_key'] and head['nemāz'][0]['source_unit_key']==sr['parent_head_key']
   targets=[t for t in targets if t['gloss']=='fajr prayer'];assert len(targets)==1
   reason='Cross-reference appears among the nemāz prayer subentries; explicit noun fajr-prayer sense, not the adverbial morning sense.'
  out=[]
  for i,t in enumerate(targets,1):
   c=copy.deepcopy(before[0]);c['entry_key']=key+f':sense:{i}';c['gloss']=t['gloss'];c['tags']=list(dict.fromkeys(t['tags']+['alternate']));c['variant_of_key']=t['entry_key'];c['status']='source relation proposed';c['notes']='Unstressed form.' if raw.startswith('=unstressed') else '';out.append(c)
  r['candidates']=out;decisions.append(dict(source_unit_key=key,raw=raw,reason=reason,targets=[t['entry_key'] for t in targets]))
 elif sr.get('parent_scope'):
  parent=next(x for x in pdf if x['source_unit_key']==sr['parent_head_key']);assert len(parent['candidates'])==1
  t=parent['candidates'][0];c=r['candidates'][0];c.update(gloss=t['gloss'],tags=list(dict.fromkeys(t['tags']+['pl'])),derivation_parent_keys=[t['entry_key']],status='source relation proposed');decisions.append(dict(source_unit_key=key,reason='Printed Plural subentry under page-spanning head.',target=t['entry_key']))
 elif sr.get('paradigm_candidates'):
  for c in r['candidates'][1:]:c['derivation_parent_keys']=[r['candidates'][0]['entry_key']];c['status']='source relation proposed'
  decisions.append(dict(source_unit_key=key,reason='Explicit complete forms in source Form field; inflection relationship to its own head only.',targets=[c['entry_key'] for c in r['candidates'][1:]]))
(P/'witness-relations-proposed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(P/'source-relation-decisions.json').write_text(json.dumps(dict(status='independent review pending',decisions=decisions),ensure_ascii=False,indent=2)+'\n')
print('physical',len(rows),'candidateforms',sum(len(r['candidates']) for r in rows),'decisions',len(decisions))
