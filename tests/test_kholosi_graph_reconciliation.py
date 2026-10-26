"""Source-supported graph repair, without replacing historical assignments."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/kholosi_2021'
R=P/'graph-reconciliation-r2'
def jsonl(name):return list(map(json.loads,(R/name).open()))
def output():return {r[10]:r for r in csv.reader((R/'whole-graph-draft.csv').open())}

def test_complete_claim_census_and_mixed_case_identifier():
 claims=jsonl('complete-source-claim-dispositions.jsonl')
 assert len(claims)==len({c['claim_key'] for c in claims})==830
 assert all(c['graph_disposition'] and c['review_status'] for c in claims)
 meo=next(c for c in claims if c['source_unit_key'].endswith('lexicon.csv:row137'))
 assert 'CDIAl 10323' in meo['literal_claim']
 assert meo['numeric_decision']['proposed_parameter_ids']==['10323']
 assert output()[meo['emitted_entry_keys'][0]][1]=='10323'

def test_direct_link_and_rejected_or_disjunctive_claim_scope():
 rows=output();decisions=jsonl('numeric-claim-decisions.jsonl')
 direct=next(d for d in decisions if d['source_unit_key'].endswith('lexicon.csv:row37'))
 assert direct['proposed_parameter_ids']==['6368']
 assert all(rows[k][1]=='6368' for k in direct['emitted_entry_keys'])
 for d in decisions:
  if d['source_form']=='xallo' or (d['source_form']=='dur'):
   assert not d['proposed_parameter_ids']
   assert all(not rows[k][1] for k in d['emitted_entry_keys'] if not k.startswith('legacy:'))
 bere=next(d for d in decisions if d['source_form']=='bere')
 assert bere['proposed_parameter_ids']==['9187'] and bere['comparison_only_ids']==['9193']
 assert all(rows[k][1]=='9187' and 'uncertain' in rows[k][14].split() for k in bere['emitted_entry_keys'])

def test_printed_number_errors_are_not_silently_treated_as_valid_heads():
 decisions=jsonl('numeric-claim-decisions.jsonl')
 for form,wrong,right in [('ānen','11714','1174'),('sino','12372','12732'),('tʰāṛgo','6386','6368'),('parono','3283','8283')]:
  selected=[d for d in decisions if d['source_form']==form and wrong in d['printed_ids']]
  assert selected
  for d in selected:
   assert d['proposed_parameter_ids']==[right]
   assert d['number_head_corrections'][0]['printed_id']==wrong
   assert d['number_head_corrections'][0]['proposed_id']==right
 for d in decisions:
  if d['source_form']=='ke':assert d['printed_ids']==['2184'] and not d['proposed_parameter_ids']

def test_explicit_component_scope_and_optional_alignment():
 rows=output();review=json.loads((R/'local-component-decisions.json').read_text())
 d=next(u for u in review if u['source_form']=='derāz kadden')
 component=d['components'][0]
 assert component['named_component']=='kadden' and len(component['accepted_entry_keys'])==1
 parent=component['accepted_entry_keys'][0]
 assert rows[parent][2:4]==['kadden','to pull']
 assert all(parent in rows[k][13].split() for k in d['emitted_entry_keys'])
 d=next(u for u in review if u['source_form']=='nevalem(te)')
 for key in d['emitted_entry_keys']:
  optional=[rows[k][2] for k in rows[key][13].split() if rows[k][2] in ('valem','valemte')]
  assert optional==[rows[key][2][2:]]
 for d in review:
  if d['source_form'] in ('hite xaten','xān xaten'):
   assert all(not c['accepted_entry_keys'] for c in d['components'])

def test_lexical_identity_historical_ancestry_and_graph_cycles():
 before=list(csv.reader((P/'whole-proposal-r1/whole-proposed.csv').open()))
 rows=output()
 assert list(rows)==[r[10] for r in before]
 for old in before:
  new=rows[old[10]]
  assert all(old[i]==new[i] for i in (0,2,3,4,5,7,8,9,10,11,12))
  if old[10].startswith('legacy:'):assert new[1]==old[1]
 def visit(key,stack):
  assert key not in stack,(key,stack)
  for parent in rows[key][13].split():
   assert parent in rows
   visit(parent,stack|{key})
 for key in rows:visit(key,set())

def test_real_parser_report_matches_current_graph_draft():
 report=json.loads((R/'actual-parser-validation.json').read_text())
 assert report['proposal_sha256']==hashlib.sha256((R/'whole-graph-draft.csv').read_bytes()).hexdigest()
 assert report['unknown_tags']==[] and report['closed_graph']
 assert all(c==dict(rows=1135,parser_errors=0) for c in report['checks'].values())
