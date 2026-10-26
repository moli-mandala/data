"""Record source-grounded component decisions, retaining unmatched boundaries."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
units=json.loads((R/'local-component-review-queue.json').read_text())
for u in units:
 pdf=':pdf:' in u['source_unit_key']
 for c in u['components']:
  candidates=c['candidates']
  c['accepted_entry_keys']=[]
  if pdf and c['named_component']=='xaten':
   c.update(disposition='unresolved sense alignment',reason='Component is glossed emit or ?; independent head means shine. Exact spelling alone does not justify a homonym/sense link.')
  elif pdf and u['source_form']=='nevalm':
   c.update(disposition='donor versus local-component scope unresolved',reason='Source wording Achomi ne + valm may scope the donor label across both parts. Preserve claim without converting donor morphology into a confident local derivation.')
  elif not candidates:
   c.update(disposition='unmatched exact source component',reason='No independent same-representation head with this exact spelling. No inferred normalization or new lexical head.')
  else:
   accepted=candidates
   if not pdf and c['named_component']=='noko':accepted=[m for m in candidates if m['form']=='noko']
   if not pdf and c['named_component']=='vāt':accepted=[m for m in candidates if m['gloss']=='path, road']
   if not pdf and c['named_component']=='māv ~ mov':
    c.update(disposition='unmatched exact source component',reason='Claim names māw; available māv/mov variants do not establish a uniquely exact cited spelling. Preserve the possible relation without silently changing w to v in Original.')
    continue
   if len(accepted)>1:
    c.update(disposition='explicit alternate component set',reason='Source names the optional component paradigm; all printed alternatives retained, without choosing one unprinted underlying stem.')
   else:
    assert len(accepted)==1,(u,c)
    c.update(disposition='accepted exact component witness',reason='The source explicitly names this component; exact same-representation head and contextual sense match.')
   c['accepted_entry_keys']=[m['entry_key'] for m in accepted]
 u['source_qualified']='probably' in u['literal_claim'].lower() or '?' in u['literal_claim']
 u['status']='reviewed explicit component claim; unresolved parts remain separately accounted'
(R/'local-component-decisions.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
print('Reviewed',len(units),'component claims; accepted parent occurrences',sum(len(c['accepted_entry_keys']) for u in units for c in u['components']))
