"""Whole-source review proposal only: never writes canonical inputs."""
import copy
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
read=lambda name:[json.loads(line) for line in (P/name).open()]
units=read('witness-metadata-reviewed-proposed.jsonl')
unit_by_key={u['source_unit_key']:u for u in units}
identity=json.loads((P/'legacy-identity-review.json').read_text())['records']
legacy=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
authority=json.loads((P/'source-metadata-plan-20260927.json').read_text())['source_authorities']
claims={c['source_unit_key']:c for c in read('source-etymology-claims.jsonl')}
second_read=json.loads((P/'owner-second-reading-decisions.json').read_text())

def locator(u):
 r=u['source_record']
 if u['representation']=='published PDF':
  return f"kholosi[p. {r['printed_page']}, col. {r['column']}, {u['source_unit_key'].rsplit(':',1)[-1]}]"
 return f"kholosi[repository 18b43f982efa, {r['source_file']}, row {r['physical_csv_row']}]"

def append_note(c,text):
 if text:
  c['notes']=' '.join(x for x in (c.get('notes','').strip(),text.strip()) if x)

for u in units:
 u['citation']=locator(u)
 r=u['source_record']
 for c in u['candidates']:
  decision=second_read['units'].get(u['source_unit_key'],{})
  if 'gloss' in decision:c['gloss']=decision['gloss']
  c['tags']+=decision.get('add_tags',[])
  append_note(c,decision.get('notes',''))
  if c['gloss']=='unknown':
   c['gloss']='';c['tags'].append('uncertain')
   append_note(c,'The source explicitly gives the meaning as unknown.')
  if c['gloss']=='first':c['tags']+=['num','ord']
  if u['representation']=='published PDF':
   if c.get('authority'):
    append_note(c,'Source attribution for this sense: '+c['authority']+'.')
   for label in c.get('source_qualifiers',[]):
    append_note(c, f"Source attribution {label}: "+authority.get('PDF '+label,label)+'.')
  else:
   if r.get('source_qualifier'):
    append_note(c,'Source citation: '+r['source_qualifier']+'.')
   if r.get('recorded'):
    append_note(c,'Source record date: '+r['recorded']+'.')
   pe=r.get('pronunciation_evidence',{})
   if pe.get('IPA 2'):
    append_note(c,'Additional pronunciation as labelled in the source: '+pe['IPA 2']+'.')
   if c.get('source_interlinear_analysis'):
    append_note(c,'Source interlinear analysis: '+c['source_interlinear_analysis']+'.')
   if c.get('comparison'):
    c['etymology']=' '.join(x for x in (c.get('etymology',''),'Source comparison: '+c['comparison']+'.') if x)
  if c.get('uncertainty_type') and 'uncertain' not in c['tags']:
   c['tags'].append('uncertain')
  c['tags']=list(dict.fromkeys(c['tags']))
 u['owner_second_reading_decision']=decision if u['candidates'] else {}

# The unstressed ne reference inherits the now-separated NEG metadata.
for u in units:
 if u['source_record'].get('cross_reference','').startswith('=unstressed'):
  c=u['candidates'][0]
  parent=next(v for v in units for t in v['candidates'] if t['entry_key']==c['variant_of_key'])
  target=parent['candidates'][0]
  c['gloss']=target['gloss'];c['tags']=list(dict.fromkeys(target['tags']+['alternate']))

# Additional explicitly quoted target attestations and complete note expressions.
for r in read('supplemental-annotation-occurrences.jsonl'):
 parent=unit_by_key[r['parent_key']]
 tags=list(r.get('tags',[]))
 if r['source_unit_key'].endswith(':quoted-anonby'):
  pos_tags={'noun','verb','adj','adv','pron','num','postp','prep','conj','interj','part','determiner'}
  tags+=sorted(set(parent['candidates'][0]['tags']) & pos_tags)
 if r.get('uncertainty_type'):
  tags.append('uncertain')
 if ' ' in r['form']:
  tags.append('multiword-expression')
 c=dict(entry_key=r['source_unit_key'],form=r['form'],gloss=r['gloss'],phonetic='',tags=tags,
        notes='Source citation: '+r.get('raw_source','')+'.',etymology='')
 if r.get('uncertainty_type'):
  append_note(c,r['uncertainty_type']+'.')
 units.append(dict(source_unit_key=r['source_unit_key'],representation='quoted target attestation in author repository',
                   source_record=r,citation=parent['citation'],candidates=[c]))

morph=json.loads((P/'bound-morphology-review-20260927.json').read_text())
for extra in second_read['additional_morphology']:
 matches=[u for u in units if u['representation']=='published PDF' and u['source_record'].get('form')==extra['parent_form'] and u['source_record']['printed_page']==extra['printed_page']]
 assert len(matches)==1
 morph['target_units'].append(dict(extra,parent_key=matches[0]['source_unit_key']))
morph['controls']+=second_read['context_only']
for n,r in enumerate(morph['target_units'],1):
 parent=unit_by_key[r['parent_key']]
 key=r['parent_key']+f':morphology:{n}'
 u=dict(source_unit_key=key,representation='explicit bound morphology',source_record=r,citation=parent['citation'],candidates=[])
 if r.get('reuse_target'):
  u['reuse_source_unit']=r['reuse_target']
 else:
  c=dict(entry_key=key,form=r['form'],gloss=r['gloss'],phonetic='',tags=r['tags'],
         notes=r.get('notes',''),etymology='')
  append_note(c,'Source analysis: '+r['source_analysis'])
  if r.get('uncertainty_type'):
   append_note(c,r['uncertainty_type']+'.')
  u['candidates']=[c]
 units.append(u)
for n,r in enumerate(morph['controls'],1):
 units.append(dict(source_unit_key=f'kholosi:morphology:control:{n}',representation='comparison or phonological control',
                   source_record=r,candidates=[],disposition='audit-only source control'))

# Source line breaks do not assert sentence boundaries. Keep this untranslated
# narrative together, without assigning invented component meanings or POS.
lines=(P/'upstream/sentences.txt').read_text().splitlines()
textkey='kholosi:github:18b43f982efa:sentences.txt:whole-text'
units.append(dict(source_unit_key=textkey,representation='untranslated source text',source_record=dict(lines=lines),
 citation='kholosi[repository 18b43f982efa, sentences.txt, lines 1–17]',candidates=[dict(
 entry_key=textkey,form=' '.join(lines),gloss='',phonetic='',tags=['multiword-expression','uncertain'],etymology='',
 notes='The source supplies this whole text without a translation. Original line boundaries and notation are retained in the source audit; no word meanings or sentence segmentation are inferred.')]))
for name in ('data.txt','data2.txt','data3.txt'):
 units.append(dict(source_unit_key='kholosi:github:18b43f982efa:'+name,representation='acoustic data control',
 source_record=dict(file=name,sha256=hashlib.sha256((P/'upstream'/name).read_bytes()).hexdigest()),
 candidates=[],disposition='Numeric formant tracks with segmented environment labels, not independently translated lexical entries.'))

candidate_by_key={c['entry_key']:c for u in units for c in u['candidates']}
assert len(candidate_by_key)==sum(len(u['candidates']) for u in units)
remap={};legacy_sources={};historical=[]
for d in identity:
 if d['disposition']=='retain historical witness separately':
  historical.append(d);continue
 target=d['entry_key']
 for sourcekey in d['equivalent_new_source_units']:
  u=unit_by_key[sourcekey]
  assert len(u['candidates'])==1
  candidate=u['candidates'][0]['entry_key']
  assert candidate not in remap
  remap[candidate]=target
  legacy_sources.setdefault(target,[]).append(u)

def emitted(c,citation,key=None,parameter='',old=None):
 notes=c.get('notes','')
 if old and old[6] and old[6] not in notes:
  notes=' '.join(x for x in (notes,'Historical Jambu witness note: '+old[6]) if x)
 return ['Khol',parameter,c['form'],c['gloss'],'',c.get('phonetic',''),notes,citation,'',c.get('etymology',''),
         key or c['entry_key'],'','','',' '.join(dict.fromkeys(c['tags']))]

output=[];output_by_key={}
for old,d in zip(legacy,identity):
 key=old[10]
 if key in legacy_sources:
  sourceunit=next(u for u in legacy_sources[key] if any(c['entry_key']==d['proposed_source_entry_key'] for c in u['candidates']))
  c=sourceunit['candidates'][0]
  row=emitted(c,';'.join(dict.fromkeys(u['citation'] for u in legacy_sources[key])),key,old[1],old)
 else:
  row=old.copy()
 output.append(row);output_by_key[key]=row
for u in units:
 u['emitted_entry_keys']=[]
 for c in u['candidates']:
  key=remap.get(c['entry_key'],c['entry_key'])
  u['emitted_entry_keys'].append(key)
  if key not in output_by_key:
   row=emitted(c,u['citation']);output.append(row);output_by_key[key]=row
  row=output_by_key[key]
  if c.get('variant_of_key'):
   target=remap.get(c['variant_of_key'],c['variant_of_key'])
   if key!=target:
    assert not row[11] or row[11]==target
    row[11]=target
  if c.get('derivation_parent_keys'):
   row[13]=' '.join(dict.fromkeys(remap.get(p,p) for p in c['derivation_parent_keys']))
 if u.get('reuse_source_unit'):
  target=unit_by_key[u['reuse_source_unit']]
  assert len(target['candidates'])==1
  key=remap.get(target['candidates'][0]['entry_key'],target['candidates'][0]['entry_key'])
  u['emitted_entry_keys']=[key]
  row=output_by_key[key]
  row[7]=';'.join(dict.fromkeys(row[7].split(';')+[u['citation']]))
 u['disposition']=u.get('disposition','emitted' if u['emitted_entry_keys'] else 'source blank')
 if u['source_unit_key'] in claims:
  u['source_etymology_claim']=claims[u['source_unit_key']]

for d in historical:
 units.append(dict(source_unit_key=d['entry_key'],representation='historical Jambu witness',source_record=d,
                   candidates=[],emitted_entry_keys=[d['entry_key']],disposition='historical witness and assignment retained'))
keys={r[10] for r in output}
assert len(keys)==len(output)
assert [r[10] for r in output[:198]]==[r[10] for r in legacy]
assert [r[1] for r in output[:198]]==[r[1] for r in legacy]
for r in output:
 assert len(r)==15 and r[2] and (not r[11] or r[11] in keys)
 assert all(k in keys for k in r[13].split())
with (P/'whole-proposed.csv').open('w') as f:
 csv.writer(f).writerows(output)
(P/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
delta=[dict(entry_key=a[10],before=a,after=b) for a,b in zip(legacy,output) if a!=b]
(P/'whole-proposed-summary.json').write_text(json.dumps(dict(status='unfrozen proposal; independent review and parser/profile validation pending',
 rows=len(output),audit_units=len(units),legacy_keys=198,historical_witnesses_retained=len(historical),
 equivalent_legacy_reuses=198-len(historical),legacy_delta=delta,
 deferred=['Canonical installation','Full CLDF/DB build','Full suite','Browser QA']),ensure_ascii=False,indent=2)+'\n')
print('proposed rows',len(output),'audit units',len(units),'legacy retained',198,'historical separate',len(historical))
