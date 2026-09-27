"""Whole-source Sikalgari stage, preserving interlinear alignment and legacy keys."""
import csv,hashlib,importlib.util,json,unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_sikalgari_1922'
spec=importlib.util.spec_from_file_location('sikalgari_full',P/'import_source_full.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

def test_whole_scope_determinism_and_all_legacy_keys():
 rows,audit=source.generate()
 assert len(rows)==633 and len(audit)==790
 assert source.generate()==(rows,audit)
 assert len({r[10] for r in rows})==633
 assert len({a['source_cell_key'] for a in audit})==790
 assert sum(a['status']=='exact_reuse' for a in audit)==157
 assert sum(a['section']=='grammar_prose' for a in audit)==86
 assert sum(a['section']=='interlinear_specimen' for a in audit)==463
 assert sum(a['section']=='translated_sentence' for a in audit)==22
 assert [int(a['unit'][5:]) for a in audit if a['unit'].startswith('item:')]==list(range(1,242))
 old=list(csv.reader((P/'legacy-installed-before-full.csv').open()))
 assert len(old)==82 and {r[10] for r in old}<={r[10] for r in rows}
 assert all(len(r)==15 and r[2] for r in rows)

def test_exact_reuse_preserves_every_source_attestation_and_gloss():
 rows,audit=source.generate();by={r[10]:r for r in rows}
 for a in audit:
  assert len(a['entry_keys'])==1
  row=by[a['entry_keys'][0]]
  assert row[2]==a['visual_reading'] and row[3]==a['english_prompt']
  assert f"{source.SOURCE}[{a['citation_locator']}]" in row[7].split(';')
  if a['unit'].startswith('item:'):assert a['status']=='ingested'
  assert a['pdf_page']==a['printed_page']+12
 assert all(not r[c] for r in rows for c in (1,4,5,8,9,11,12,13))

def test_high_resolution_literal_corrections_and_uncertainty():
 rows,audit=source.generate();by={(a['printed_page'],a['unit']):a for a in audit}
 expected={(181,'item:12'):'Bē-ikh-dakh',(193,'item:80'):'Āk̲h̲ṭal',(193,'item:100'):'Ayyᵃyyō',(197,'item:130'):'Chōkīyō bākḍiyō',(167,'got'):'maḷyū',(167,'searched'):'sādīnē',(168,'ch-substitution-way'):'chāyē',(170,'specimen:line4:word6'):'līne',(171,'specimen:line9:word1'):'lāvīne',(172,'specimen:line2:word8'):'duṭwā',(173,'specimen:line4:word3'):'to'}
 for key,value in expected.items():assert by[key]['visual_reading']==unicodedata.normalize('NFC',value)
 uncertain=[a for a in audit if a['typed_uncertainty']]
 assert len(uncertain)==4
 assert all('uncertain' in a['source_tags'].split() for a in uncertain)
 assert by[181,'item:9']['visual_reading']=='Ṇau'
 assert by[185,'item:40']['visual_reading']==unicodedata.normalize('NFC','Mạ̄tū')
 assert by[213,'item:235']['visual_reading'].startswith('Ti-kántā')
 assert by[173,'specimen:line1:word3']['visual_reading']=='khyāpāryō'
 assert 'v reading' in by[173,'specimen:line1:word3']['typed_uncertainty']

def test_profile_coverage_and_literal_symbol_layer():
 rows,_=source.generate();t=Tokenizer(str(P/'full-profile-staged.txt'))
 for r in rows:
  assert unicodedata.normalize('NFC',r[2])==r[2]
  assert '�' not in t(r[2],column='IPA')
 for form in ['Ṇau','Mạ̄tū','Ti-kántā','Āk̲h̲ṭal','t̲s̲ākrī','Ayyᵃyyō','nikartaū̃']:
  assert '�' not in t(unicodedata.normalize('NFC',form),column='IPA')
 assert t('thauṅgā',column='IPA').replace(' ','')=='thauŋgā'

def test_source_construction_metadata_and_qualified_dialect():
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 rows,audit=source.generate();by={int(a['unit'][5:]):a for a in audit if a['unit'].startswith('item:')}
 for r in rows:
  assert source.DIALECT in r[14].split()
  assert set(r[14].split())-{source.DIALECT}<=GRAMMATICAL_TAGS|GENDER_TAGS
 for n in range(156,220):assert 'verb' in by[n]['source_tags'].split()
 for n in range(168,220):assert 'Source construction label:' in by[n]['source_notes']
 for n,tags in {191:{'pres','progressive','1sg'},193:{'pret','perfect','1sg'},204:{'fut','pass','1sg'}}.items():assert tags<=set(by[n]['source_tags'].split())
 languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
 assert 'Sik' in languages
 dialects=list(csv.DictReader((DATA/'cldf/dialects.csv').open()))
 assert any(r['Language_ID']=='Sik' and r['ID']=='sik_belgaum' for r in dialects)

def test_original_specimens_and_translation_preserved():
 manifest=json.loads((P/'specimen-source-preservation.json').read_text())
 assert hashlib.sha256((P/manifest['witness']).read_bytes()).hexdigest()==manifest['witness_sha256']
 assert [x['printed'] for x in manifest['pages']]==[170,171,172,173,174]
 assert manifest['bytes']<2*1024*1024

def test_scoped_preview_parser_preserves_originals_and_keys(monkeypatch):
 import io,make_cldf
 rows,_=source.generate();raw={r[10]:r for r in rows}
 monkeypatch.setitem(make_cldf.convertors,'grierson-sikalgari-1922',Tokenizer(str(P/'full-profile-staged.txt')))
 errors=io.StringIO()
 parsed,stats=make_cldf.parse_file(str(P/'full-staged.csv'),errors,name='20260925-grierson-sikalgari')
 assert not errors.getvalue() and len(parsed)==stats['converted']==633
 assert {r.entry_key for r in parsed}==raw.keys()
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
