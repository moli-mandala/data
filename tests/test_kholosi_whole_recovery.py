"""Source-grounded Kholosi recovery checks; no canonical or compiled writes."""
import csv
import hashlib
import json
import unicodedata as ud
from pathlib import Path

from segments import Tokenizer

D=Path(__file__).resolve().parents[1]
P=D/'data/other/forms/raw_data/kholosi_2021'

def read(name):
 return [json.loads(line) for line in (P/name).open()]

def rows():
 return list(csv.reader((P/'whole-proposed.csv').open()))

def test_every_published_and_structured_physical_unit_is_accounted():
 raw=read('pdf-first-reading.jsonl')+read('structured-first-reading.jsonl')
 assert len(raw)==518+401+116
 audit={r['source_unit_key']:r for r in read('whole-proposed-audit.jsonl')}
 output={r[10]:r for r in rows()}
 assert {r['source_unit_key'] for r in raw}<=audit.keys()
 for r in raw:
  a=audit[r['source_unit_key']]
  if r.get('status')=='blank':
   assert not any(r['raw_fields']) and not a['emitted_entry_keys']
  else:
   assert a['emitted_entry_keys'],r['source_unit_key']
   assert set(a['emitted_entry_keys'])<=output.keys()
 assert {'aber','avval','barg'}<={r[2] for r in output.values()}

def test_all_legacy_keys_and_ancestry_assignments_survive_without_homonym_guessing():
 old=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
 current=rows()
 assert len(old)==198
 assert [r[10] for r in current[:198]]==[r[10] for r in old]
 assert [r[1] for r in current[:198]]==[r[1] for r in old]
 report=json.loads((P/'legacy-identity-review.json').read_text())
 assert report['baseline_sha256']==hashlib.sha256((P/'legacy-baseline/20220913-kholosi.csv').read_bytes()).hexdigest()
 decisions={r['entry_key']:r for r in report['records']}
 for n in (21,22,26,70,98,127,138,139,156,157,158,187,188):
  key=f'legacy:20220913-kholosi:row:{n}'
  assert decisions[key]['disposition']=='retain historical witness separately'
  assert current[n-1]==old[n-1]
 # The full compound is present independently of the historical isolated analysis.
 assert any(r[2]=='puni jelji' and r[10]!=old[126][10] for r in current)

def test_etymon_conflicts_preserve_printed_qualifier_and_editorial_assignment():
 claims={c['source_unit_key']:c for c in read('source-etymology-claims.jsonl')}
 for key,n,source_id in [('kholosi:pdf:p1:c1:u24',9,'11714'),('kholosi:pdf:p8:c1:u7',147,'12415'),('kholosi:pdf:p8:c1:u19',154,'12372')]:
  c=claims[key];conflict=c['legacy_conflict']
  assert conflict['entry_key']==f'legacy:20220913-kholosi:row:{n}'
  assert conflict['printed_target_id']==source_id
  assert conflict['historical_jambu_parameter_id']!=source_id
  assert all(t['exists'] for t in c['cited_target_ids'])
 assert 'probably' in claims['kholosi:pdf:p8:c1:u7']['qualifier_words']

def test_source_metadata_does_not_leak_feminine_and_unknown_is_not_a_definition():
 audit={a['source_unit_key']:a for a in read('whole-proposed-audit.jsonl')}
 output={r[10]:r for r in rows()}
 for n in (7,8,9,10,11,12,13,398):
  a=audit[f'kholosi:github:18b43f982efa:lexicon.csv:row{n}']
  assert all('f' not in output[k][14].split() for k in a['emitted_entry_keys'])
 a=audit['kholosi:github:18b43f982efa:lexicon.csv:row393']
 assert all('f' in output[k][14].split() for k in a['emitted_entry_keys'])
 for key in ('kholosi:pdf:p3:c1:u21','kholosi:pdf:p3:c2:u15'):
  for k in audit[key]['emitted_entry_keys']:
   assert output[k][3]=='' and 'uncertain' in output[k][14].split()
 for n in (28,32,49,50):
  for k in audit[f'kholosi:github:18b43f982efa:sentences.csv:row{n}']['emitted_entry_keys']:
   assert 'uncertain' in output[k][14].split()
   assert 'Source interlinear analysis:' in output[k][6]

def test_explicit_alternates_morphology_and_long_text_remain_complete():
 audit={a['source_unit_key']:a for a in read('whole-proposed-audit.jsonl')}
 output={r[10]:r for r in rows()}
 forms=lambda key:{output[k][2] for k in audit[key]['emitted_entry_keys']}
 assert forms('kholosi:github:18b43f982efa:lexicon.csv:row44')=={'māv','mov'}
 assert forms('kholosi:github:18b43f982efa:lexicon.csv:row46')=={'pu','pʰu'}
 assert forms('kholosi:github:18b43f982efa:lexicon.csv:row47')=={'viyāv','viyov'}
 assert {'pen','pʰen'}==forms('kholosi:github:18b43f982efa:lexicon.csv:row141')
 assert any(r[2]=='h-' and {'prefix','emph'}<=set(r[14].split()) for r in output.values())
 k='kholosi:github:18b43f982efa:sentences.txt:whole-text'
 lines=(P/'upstream/sentences.txt').read_text().splitlines()
 assert len(lines)==17 and audit[k]['source_record']['lines']==lines
 assert output[k][2]==' '.join(lines) and output[k][3]==''
 assert 'sentential' not in output[k][14].split()
 assert all(not r[11] or r[11] in output for r in output.values())
 assert all(k in output for r in output.values() for k in r[13].split())

def test_profile_covers_every_literal_and_actual_parser_report_is_current():
 profile=P/'whole-profile-proposed.txt'
 tokenizer=Tokenizer(str(profile))
 for row in rows():
  values=[ud.normalize('NFC',tokenizer(ud.normalize(n,row[2]),column='IPA')) for n in ('NFC','NFD')]
  assert values[0]==values[1] and '�' not in values[0],row[10]
 report=json.loads((P/'whole-parser-validation.json').read_text())
 assert report['proposal_sha256']==hashlib.sha256((P/'whole-proposed.csv').read_bytes()).hexdigest()
 assert report['profile_sha256']==hashlib.sha256(profile.read_bytes()).hexdigest()
 assert report['unknown_tags']==[] and report['closed_graph']
 assert all(c['rows']==len(rows()) and c['parser_errors']==0 for c in report['checks'].values())
