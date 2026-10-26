"""Source-local semantic witness draft. Not a canonical importer or approved freeze."""
import csv,json,re,hashlib,collections
from pathlib import Path
P=Path(__file__).resolve().parent
pdf=[json.loads(l) for l in (P/'pdf-first-reading.jsonl').open()]
meta=json.load((P/'pdf-metadata-decisions-20260927.json').open())
records=[]
for r in pdf:
 key=r['source_unit_key'];base={k:r.get(k,'') for k in ('form','gloss','phonetic','etymology')};base.update(notes='',tags=list(r['tags']),source_qualifiers=r.get('source_qualifiers',[]))
 decision=meta['units'].get(key,{})
 for f in ('gloss','notes','uncertainty_type'):
  if f in decision:base[f]=decision[f]
 base['tags']=list(dict.fromkeys(decision.get('replace_tags',base['tags'])+decision.get('add_tags',[])))
 if r.get('expression_classification_pending'):
  base['tags'].append('multiword-expression' if r['form'] in ('bere noko','hinki sānda') else 'sentential')
  if r['form'].endswith('?'):base['tags'].append('interr')
 if ' ' in base['form']:base['tags'].append('multiword-expression')
 candidates=[]
 for j,sense in enumerate(meta['mixed_sense_units'].get(key,[{}]),1):
  c=dict(base);c.update(sense);c['entry_key']=key+f':sense:{j}';c['tags']=list(dict.fromkeys(c['tags']));candidates.append(c)
 if r.get('cross_reference'):
  for c in candidates:c['cross_reference']=r['cross_reference'];c['status']='pending exact cross-reference sense resolution'
 else:
  for c in candidates:c['status']='candidate'
 if r.get('parent_scope'):
  for c in candidates:c['status']='pending parent-gloss inheritance';c['parent_head_key']=r['parent_head_key']
 records.append(dict(source_unit_key=key,representation='published PDF',source_locator=f"kholosi[p. {r['printed_page']}, col. {r['column']}, {key.rsplit(':',1)[-1]}]",source_record=r,metadata_decision=decision,candidates=candidates,status='editorial review pending'))
for r in map(json.loads,(P/'structured-first-reading.jsonl').open()):
 cs=[]
 if r['status']=='target':
  c={k:r.get(k,'') for k in ('form','gloss','phonetic','etymology','comparison','notes','tags')};c['entry_key']=r['source_unit_key'];c['tags']=list(c['tags'])
  if ' ' in c['form']:c['tags'].append('multiword-expression')
  c['status']='pending alternate segmentation' if r.get('alternate_segmentation_pending') else 'candidate';cs.append(c)
  for q in r.get('paradigm_candidates',[]):
   cs.append(dict(entry_key=q['source_unit_key'],form=q['form'],gloss=q['gloss'],phonetic='',tags=q['tags'],notes=q['notes'],status='pending source inflection-link review'))
 records.append(dict(source_unit_key=r['source_unit_key'],representation='author repository CSV',source_record=r,candidates=cs,status='blank' if r['status']=='blank' else 'editorial review pending'))
(P/'witness-proposed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print('physical_units',len(records),'candidate_witnesses',sum(len(x['candidates']) for x in records),'states',collections.Counter(c['status'] for x in records for c in x['candidates']))
