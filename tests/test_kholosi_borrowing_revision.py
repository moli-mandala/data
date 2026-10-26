"""Regressions for the independently observed borrowing and number errors."""
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/kholosi_2021'
R=P/'graph-reconciliation-r3'
def rows():return {r[10]:r for r in csv.reader((R/'whole-proposed.csv').open())}

def test_explicit_persian_attributions_and_variants_are_classified():
 out=rows();review=json.loads((R/'donor-classification-proposal.json').read_text())
 audit=list(map(json.loads,(P/'whole-proposal-r2/whole-proposed-audit.jsonl').open()))
 for form in ('dāno','derāz','xorāk','kavi'):
  units=[u for u in audit if u['representation']=='published PDF' and u['source_record'].get('form')==form]
  assert units,form
  for u in units:
   assert u['source_record']['etymology'].startswith('Persian')
   assert all('loanword' in out[k][14].split() for k in u['emitted_entry_keys'])
 for key,row in out.items():
  if key.endswith('lexicon.csv:row71') or (row[2]=='valem' and row[11]):
   assert 'loanword' in row[14].split()
 assert len(review['claims'])==830
 assert all('loanword' in out[k][14].split() for k in review['row_classifications'])

def test_comparisons_and_blends_do_not_become_unqualified_whole_loans():
 out=rows();review=json.loads((R/'donor-classification-proposal.json').read_text())
 for c in review['claims']:
  if ':pdf:' in c['source_unit_key'] and c['literal_claim'].startswith(('Persian dūr or','Persian sūt +','cf. Koroshi')):
   assert not c['loanword']
   assert all('loanword' not in out[k][14].split() for k in c['emitted_entry_keys'])
 for c in review['claims']:
  if c['literal_claim']=='perhaps Persian pēš.':
   assert c['loanword'] and c['qualified']
   assert all({'loanword','uncertain'}<=set(out[k][14].split()) for k in c['emitted_entry_keys'])
 byunit={c['source_unit_key']:c for c in review['claims']}
 for key in ('kholosi:pdf:p4:c2:u7','kholosi:pdf:p10:c1:u11'):
  assert byunit[key]['loanword']
 for key in ('kholosi:github:18b43f982efa:lexicon.csv:row86','kholosi:github:18b43f982efa:lexicon.csv:row329',
             'kholosi:pdf:p6:c1:u21','kholosi:pdf:p6:c1:u22','kholosi:pdf:p6:c1:u23',
             'kholosi:github:18b43f982efa:lexicon.csv:row153'):
  assert not byunit[key]['loanword']

def test_historical_wrong_heads_correct_without_changing_identity():
 out=rows();before={r[10]:r for r in csv.reader((P/'whole-proposal-r2/whole-proposed.csv').open())}
 expected={92:'9735',194:'3138',141:'12357',169:'13744',64:''}
 changes={k:(before[k][1],r[1]) for k,r in out.items() if r[1]!=before[k][1]}
 assert set(changes)=={f'legacy:20220913-kholosi:row:{n}' for n in expected}
 for n,param in expected.items():
  key=f'legacy:20220913-kholosi:row:{n}'
  assert out[key][1]==param
  assert all(out[key][i]==before[key][i] for i in (0,2,3,4,5,7,8,9,10,11,12,13))
 for n,param in ((63,'3069'),(116,'7747'),(147,'4749')):
  assert out[f'legacy:20220913-kholosi:row:{n}'][1]==param
 assert 'uncertain' in out['legacy:20220913-kholosi:row:64'][14].split()
 assert 'uncertain' in out['legacy:20220913-kholosi:row:141'][14].split()

def test_actual_graph_parser_and_frozen_lexical_continuity():
 out=rows();before=list(csv.reader((P/'whole-proposal-r2/whole-proposed.csv').open()))
 assert list(out)==[r[10] for r in before]
 for old in before:
  assert all(old[i]==out[old[10]][i] for i in (0,2,3,4,5,7,8,9,10,11,12,13))
 report=json.loads((R/'actual-parser-validation.json').read_text())
 assert report['proposal_sha256']==hashlib.sha256((R/'whole-proposed.csv').read_bytes()).hexdigest()
 assert all(c['rows']==1135 and c['parser_errors']==0 and c['exact_graph_fields'] for c in report['checks'].values())
