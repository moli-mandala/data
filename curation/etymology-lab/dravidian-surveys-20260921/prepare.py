"""Reproduce scoped, manually specified proposals; never writes accepted sidecars."""
import csv,gzip,json,re,unicodedata,collections,sys
from pathlib import Path
P=Path(__file__).resolve().parent

from research_common import fold, gloss

sys.path.insert(0,str(P.parents[2]))
from etymology_assignments import read_assignments
inventory=json.load(open(P/'survey-inventory.json'))
current_assignments=read_assignments()
assigned={r['Form_ID'] for r in current_assignments}
resolved=[r for r in inventory if r['ID'] in assigned]
(P/'externally-assigned-records.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2)+'\n')
rows=[r for r in inventory if r['ID'] not in assigned]
families=json.load(open(P/'families.json'))
direct=json.load(open(P/'direct-reviewed.json')) if (P/'direct-reviewed.json').exists() else {}
held_rules=json.load(open(P/'held-reviewed.json')) if (P/'held-reviewed.json').exists() else {}
compound_rules=json.load(open(P/'compound-rules.json')) if (P/'compound-rules.json').exists() else []
relation_rules=json.load(open(P/'relation-rules.json')) if (P/'relation-rules.json').exists() else []
compound_parents={p for rule in compound_rules for p in rule['components']}|{r['evidence_parent'] for r in relation_rules}
articles={r['Entry_ID']:r for r in csv.DictReader(gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt')) if r.get('Status') != 'duplicate-excluded' and r['Entry_ID'] in {f['parent'] for f in families}|set(direct)|set(held_rules)|compound_parents}
selected=[];held=[]
for f in families:
 allowed={fold(x) for x in f['forms'].split()}
 for r in rows:
  if gloss(r['Gloss']) not in f['glosses'] or fold(r['Form']) not in allowed or (f.get('languages') and r['Language_ID'] not in f['languages']) or r['Language_ID'] in f.get('exclude_languages',[]):continue
  reason=None
  if f['parent']=='d1977':reason='DEDR separates ear subsections (a) and (b), while existing node d1977 carries the *kir- reconstruction of (b). A precise parent for subsection (a) must be resolved before saving.'
  if f['parent']=='d3690':reason='The primary water article is correct, but existing node d3690 is headed *kaṇ-ṇīr tears. Resolve the water-versus-compound parent label before saving new water reflexes.'
  if f['parent']=='d651':reason='The primary mortar article is correct, but existing node d651 is headed *uram-kkal grinding stone/pestle. Resolve this parent-headword mismatch before saving.'
  if f['parent']=='d4985' and r['Language_ID']=='Kota' and fold(r['Form'])=='mol':reason='DEDR expressly marks Kota mol as a Badaga loan; immediate donor requires resolution.'
  if f['parent']=='d1204' and r['Language_ID']=='Gondi':reason='DEDR Gondi attestation has specific cock-spur/fighting-knife meaning; contact route unresolved.'
  if f['parent']=='d5352' and fold(r['Form'])=='koy':reason='Not the documented mouth reflex; potentially unrelated.'
  if r['Language_ID']=='Gondi' and f['parent'] in {'d5024','d3633','d4085','d784'}:reason='This response resembles a neighbouring Telugu or other regional form; an inherited reflex versus borrowing requires source-local evidence.'
  if r['Language_ID']=='Gadaba' and f['parent'] in {'d2826','d2485','d910','d784','d3918','d3729'}:reason='Higher numeral resembles Telugu; the immediate-donor versus inherited analysis has not been established for this survey lect.'
  if r['Language_ID']=='Konda' and f['parent'] in {'d2826','d2485','d784','d3918','d3729'}:reason='Higher numeral resembles Telugu; source-local loan evidence and an immediate donor are needed before inheritance is asserted.'
  if r['Language_ID']=='Toda' and f['parent'] in {'d1159','d5352','d1479','d5535','d3754','d4711','d3650','d5152','d910','d4096','d3633','d1977'}:reason='Survey form needs reconciliation with the distinctive Toda reflex printed by DEDR; possible contact, source-label or transcription issue.'
  if r['Language_ID']=='Toda' and f['parent']=='d1514' and fold(r['Form']) in {'kic','kiccu'}:reason='DEDR identifies Toda song kic as borrowed from Badaga; needs an immediate-donor link.'
  if reason:held.append({'record':r,'candidate':f['parent'],'reason':reason});continue
  selected.append({'record':r,'parent':f['parent'],'kind':'reflex','citation':f"dedr[entry {f['parent'][1:]}, p. {articles[f['parent']]['Page']}]",'evidence':f['evidence'],'status':'pending-review'})
seen={x['record']['ID'] for x in selected}|{x['record']['ID'] for x in held}
if direct:
 candidates=json.load(open(P/'direct-candidates.json'))
 for parent,rule in direct.items():
  for x in candidates.get(parent,[]):
   r=x['record']
   if r['ID'] in assigned or r['ID'] in seen or (rule.get('glosses') and gloss(r['Gloss']) not in rule['glosses']) or (rule.get('languages') and r['Language_ID'] not in rule['languages']) or (rule.get('forms') and fold(r['Form']) not in {fold(v) for v in rule['forms']}):continue
   selected.append({'record':r,'parent':parent,'kind':'reflex','citation':f"dedr[entry {parent[1:]}, p. {articles[parent]['Page']}]" + (';' + rule['additional_citation'] if rule.get('additional_citation') else ''),'evidence':rule['evidence'],'comparison':x['comparison'],'status':'pending-review'})
   seen.add(r['ID'])
 for parent,reason in held_rules.items():
  for x in candidates.get(parent,[]):
   r=x['record']
   if r['ID'] in assigned or r['ID'] in seen:continue
   held.append({'record':r,'candidate':parent,'reason':reason,'comparison':x['comparison']})
   seen.add(r['ID'])
parent_holds=json.load(open(P/'parent-holds.json')) if (P/'parent-holds.json').exists() else {}
kept=[]
for x in selected:
 if x['parent'] in parent_holds and not (isinstance(parent_holds[x['parent']],dict) and fold(x['record']['Form']) in {fold(v) for v in parent_holds[x['parent']].get('except_forms',[])}):
  hold_rule=parent_holds[x['parent']]
  held.append({'record':x['record'],'candidate':x['parent'],'reason':hold_rule['reason'] if isinstance(hold_rule,dict) else hold_rule,'priorProposal':x})
 else:kept.append(x)
selected=kept
by_id={r['ID']:r for r in rows}
compound_ids=set()
for rule in compound_rules:
 for record_id in rule['record_ids']:
  if record_id not in by_id:continue
  assert record_id not in {x['record']['ID'] for x in selected}, record_id
  components=rule['components']
  selected.append({'record':by_id[record_id],'parent':components[0],'components':components,'kind':'component','citation':';'.join(f"dedr[entry {parent[1:]}, p. {articles[parent]['Page']}]" for parent in components),'evidence':rule['evidence'],'status':'pending-review'})
  compound_ids.add(record_id)
held=[x for x in held if x['record']['ID'] not in compound_ids]
relation_ids=set()
for rule in relation_rules:
 for record_id in rule['record_ids']:
  if record_id not in by_id:continue
  assert record_id not in {x['record']['ID'] for x in selected}, record_id
  selected.append({'record':by_id[record_id], 'parent':rule['parent'], 'evidence_parent':rule['evidence_parent'], 'kind':rule['kind'], 'citation':rule['citation'], 'evidence':rule['evidence'], 'status':'pending-review'})
  relation_ids.add(record_id)
held=[x for x in held if x['record']['ID'] not in relation_ids]
scoped_holds=json.load(open(P/'scoped-holds.json')) if (P/'scoped-holds.json').exists() else []
for rule in scoped_holds:
 for record_id in rule['record_ids']:
  if record_id not in by_id:continue
  assert record_id not in {x['record']['ID'] for x in selected+held},record_id
  held.append({'record':by_id[record_id], 'candidate':rule['candidate'], 'reason':rule['reason']})
# Explicit reviewed amendments after the user's article-level DEDR scope clarification.
# These are pending research decisions; canonical save validation remains unchanged.
scope_path=P/'article-scope-reviewed.json'
if scope_path.exists():
 scope_reviews={x['record']['ID']:x for x in json.load(open(scope_path))}
 still_held=[]
 for x in held:
  amended=scope_reviews.get(x['record']['ID'])
  if amended:
   selected.append({**amended,'record':x['record'],'status':'pending-review'})
  else:still_held.append(x)
 held=still_held
assert len({x['record']['ID'] for x in selected+held})==len(selected)+len(held), 'Duplicate research record'
by=collections.defaultdict(list)
for x in selected:by[x['parent']].append(x)
lines=[]
for f in families:
 rr=by[f['parent']];lines.append(f"\n{f['parent']} {','.join(f['glosses'])} {len(rr)} records")
 for l in sorted({x['record']['Language_ID'] for x in rr}):
  c=collections.Counter(x['record']['Form'] for x in rr if x['record']['Language_ID']==l)
  lines.append(l+': '+', '.join(f'{k} ({v})' for k,v in c.items()))
(P/'selection-preview.txt').write_text('\n'.join(lines)+'\n')
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':selected,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print('proposals',len(selected),'families',len({p for x in selected for p in x.get('components',[x['parent']])}),'holds',len(held))
