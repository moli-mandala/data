"""Installed whole Adiya source contract; no full compiled dataset required."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
from pybtex.database import parse_file

import make_cldf
import source_meta

DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/nair_adiya_2013'
STEM='20260927-nair-adiya'
CSV=DATA/f'data/other/forms/{STEM}.csv'
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
def rows():return list(csv.reader(CSV.open(newline='')))

def test_complete_source_accounting_and_exact_approved_rows():
 r=rows();a=[json.loads(x) for x in (PACKAGE/'installed-audit.jsonl').read_text().splitlines()]
 assert CSV.read_bytes()==(PACKAGE/'proposed.csv').read_bytes()
 assert len(r)==171 and len(a)==29
 assert sum(len(u['physical_witnesses']) for u in a)==174
 assert sum(u['held_or_excluded_count'] for u in a)==3
 assert sum(x[0]=='Ravula' for x in r)==136
 assert {x[10] for x in r}=={k for u in a for k in u['entry_keys']}
 assert all(u['accepted_for_source_stage'] for u in a)

def test_real_registered_parser_exact_all_fifteen_fields():
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name=STEM)
 assert not errors.getvalue()
 assert stats=={'converted':0,'for_conversion':0}
 assert len(parsed)==171
 for p,r in zip(parsed,rows()):
  assert [getattr(p,x) for x in FIELDS]==r,r[10]
  assert p.form==r[2]

def test_source_literals_and_scoped_bound_metadata():
 r={x[10]:x for x in rows()};forms={x[2] for x in r.values()}
 assert {'kal̲uttŭ','n̲aanu','enaakku par̲r̲aa','enaakku muun̲ru puḷḷe uḷa','niinŋka','niiŋka','a','–kaḷ','–va','–ma','–mi','–i','–kaaṇi','–a'}<=forms
 for key in ['nair2013adiya:p5:plural-morphology:slot1','nair2013adiya:p5:plural-morphology:slot4','nair2013adiya:p6:affinal-address-mi:slot1']:
  assert 'suffix' in r[key][14].split() and 'sentential' not in r[key][14].split()
 assert all(not x[4] and not x[5] for x in r.values())

def test_twelve_approved_variants_and_registry_reference_scope():
 r={x[10]:x for x in rows()};g=json.loads((PACKAGE/'graph-approved.json').read_text())['proposals']
 assert len(g)==sum(bool(x[11]) for x in r.values())==12
 for edge in g:
  assert r[edge['child_key']][11]==edge['variant_of_key']
  assert r[edge['child_key']][2]==edge['child_form']
  assert r[edge['variant_of_key']][2]==edge['head_form']
 meta=source_meta.load()
 assert meta.transcription('nair2013adiya',str(CSV),'Ravula')==(None,False)
 for key in ['nair2013adiya','sankarankutty1978adiya','syama1978adiya','sarala1978adiya','davidson1978adiya']:
  assert meta.flag(key,'reference','editor')=='OpenAI Codex'
  assert meta.flag(key,'reference','ocr') is False
  assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013adiya' else 'none')
 bib=parse_file(DATA/'cldf/sources.bib');assert {r[7].split('[',1)[0] for r in rows()}<=bib.entries.keys()
 dialect=[x for x in csv.DictReader((DATA/'cldf/dialects.csv').open()) if x['ID']=='nair2013_adiya_wayanad']
 assert len(dialect)==1 and dialect[0]['Language_ID']=='Ravula'
 assert dialect[0]['Glottocode']==dialect[0]['Latitude']==dialect[0]['Longitude']==''
 langs={x['ID']:x for x in csv.DictReader((DATA/'cldf/languages.csv').open())}
 assert {x[0] for x in rows()}<=langs.keys() and langs['Ravula']['Glottocode']=='ravu1237'

def test_guarded_regeneration_receipt_and_rejection_of_input_drift(monkeypatch):
 spec=importlib.util.spec_from_file_location('adiya_installer',PACKAGE/'import_source.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 result=mod.main(False);assert result['canonical_mutations'] is False
 assert result['status'].startswith('already installed')
 old=mod.file_hash
 monkeypatch.setattr(mod,'file_hash',lambda p:'bad' if p==PACKAGE/'proposed.csv' else old(p))
 with pytest.raises(AssertionError,match='approved input drift'):mod.validate_inputs()
