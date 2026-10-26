"""Source-scoped whole-book output accounting; no canonical mutation."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/niranjan_chakma_2010'
sys.path.insert(0,str(P/'whole-installation'))
spec=importlib.util.spec_from_file_location('chakma_whole_export',P/'whole-installation/export_regeneration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture(scope='module')
def exported(): return m.prepare()

def test_every_inventory_record_has_an_explicit_disposition(exported):
 rows,audit,status=exported
 inventory,_=m.build()
 assert len(audit)==len(inventory)==5232
 assert {r['source_unit_key'] for r in audit}=={r['key'] for r in inventory}
 emitted=[k for r in audit for k in r['emitted_entry_keys']]
 assert len(emitted)==len(set(emitted))==len(rows)
 assert set(emitted)=={r[10] for r in rows}
 assert all(r['reason'] for r in audit if not r['emitted_entry_keys'])
 assert all(len(r)==15 and r[2] and r[10] for r in rows)

def test_legacy_identity_and_fields_only_change_with_review_evidence(exported):
 rows,_,_=exported;old=list(csv.reader(m.CANONICAL.open()))
 assert [r[10] for r in rows[:742]]==[r[10] for r in old]
 inventory,_=m.build();by={r['key']:r for r in inventory}
 for before,after in zip(old,rows):
  assert [v for i,v in enumerate(before) if i not in (2,4,6,14)]==[v for i,v in enumerate(after) if i not in (2,4,6,14)]
  assert after[2]==after[4]==by[before[10]]['preview_reading']
  assert set(before[14].split())<=set(after[14].split())

def test_numbered_controls_and_documentary_are_not_lexical(exported):
 rows,audit,_=exported;keys={r[10] for r in rows};a={r['source_unit_key']:r for r in audit}
 decisions=json.loads((P/'numbered-identity-root-adjudication-20260927.json').read_text())['decisions']
 for d in decisions:
  k=d['source_unit_key']
  if d['accepted_language_id'] is not None or 'Chenchu' in d['disposition']: assert k in keys
  else: assert k not in keys and a[k]['reason']
 assert all(not r['emitted_entry_keys'] for r in audit if r['section'] in ('abbreviations','frontmatter','documentary'))
 for r in audit:
  if r['kind']=='source_claim': assert r['source_analysis'] and not r['emitted_entry_keys']

def test_approved_uncertainties_and_new_languages_survive(exported):
 rows,_,_=exported;by={r[10]:r for r in rows}
 inventory,_=m.build()
 for r in inventory:
  if r['key'] in by and 'uncertain' in r.get('required_tags',[]):
   assert 'uncertain' in by[r['key']][14].split() and by[r['key']][6]
 proposal=json.loads((P/'language-registry-proposal.json').read_text())
 for r in proposal['routes']: assert by[r['source_unit_key']][0]==r['proposed_base_id']
 assert sum(r[0]=='Tanchangya' for r in rows)==3
 assert sum(r[0]=='Chenchu' for r in rows)==4
