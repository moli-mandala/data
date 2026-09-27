"""Source-stage invariants for the whole LSI Simla Siraji recovery (no DB build)."""
import csv,importlib.util,json
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_simla_siraji_1916'
spec=importlib.util.spec_from_file_location('simla_full',P/'import_source_full.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

def test_whole_scope_and_stable_legacy_keys():
 rows,audit=source.generate()
 assert len(rows)==414 and len(audit)==341
 assert source.generate()==(rows,audit)
 assert len({r[10] for r in rows})==414
 old=list(csv.reader((P/'legacy-installed-before-full.csv').open()))
 assert len(old)==16 and {r[10] for r in old}<={r[10] for r in rows}
 assert [int(a['unit'].split(':')[1]) for a in audit if a['unit'].startswith('item:')]==list(range(1,242))
 assert [a['unit'] for a in audit if not a['entry_keys']]==['item:174','item:201']
 assert len([a for a in audit if a['section']=='translated_sentence'])==22
 assert len([a for a in audit if a['section']=='parable_gloss'])==5

def test_literal_marks_alternatives_and_source_qualifications():
 rows,audit=source.generate();by={r[10]:r for r in rows};base='grierson1916simlasiraji:'
 assert by[base+'p631:item:36'][2]=='Mū̃'
 assert by[base+'p631:item:41'][2]=='Jīb'
 assert by[base+'p641:item:175'][2]=='Pīṭ'
 assert by[base+'p631:item:49:answer2'][2]=='bhāī'
 assert by[base+'p635:item:96'][2]=='Sidhō'
 assert by[base+'p635:item:97'][2]=='Jai'
 assert by[base+'p641:item:178'][2]=='Pīṭĕ-rō'
 assert by[base+'p641:item:180'][2]=='Tū pīṭē'
 assert 'Printed English prompt' in by[base+'p637:item:117'][6]
 assert 'qualifies' in by[base+'p593:considered'][6]
 assert 'shared' in by[base+'p639:item:156'][6]
 assert 'Tūē̃ ḍēwē'==by[base+'p645:item:215'][2]

def test_literal_profile_and_original_text_preservation():
 rows,_=source.generate();t=Tokenizer(str(P/'full-profile-staged.txt'))
 for r in rows:assert '�' not in t(r[2],column='IPA')
 import hashlib
 manifest=json.loads((P/'parable-source-preservation.json').read_text())
 assert hashlib.sha256((P/manifest['witness']).read_bytes()).hexdigest()==manifest['witness_sha256']
 assert [p['printed'] for p in manifest['pages']]==[596,597,598]
 assert manifest['bytes']<1024*1024

def test_explicit_source_heading_metadata_complete():
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 rows,_=source.generate();by={r[10]:r for r in rows};base='grierson1916simlasiraji:'
 for r in rows:
  assert set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS
  key=r[10]
  if ':pronoun-' in key:assert {'pron','personal'}<=set(r[14].split())
  if ':demonstrative-' in key:assert {'pron','demonstrative'}<=set(r[14].split())
  if ':be-' in key:assert {'verb','copula'}<=set(r[14].split())
  if ':beat-' in key:assert 'verb' in r[14].split()
 for unit in ['rejoicing','property','cultivation']:
  assert {'noun','f'}<=set(by[base+'p593:'+unit][14].split())
 assert {'pron','personal','1sg'}<=set(by[base+'p599:simla-by-me-comparison'][14].split())
 assert 'agent' in by[base+'p599:simla-by-me-comparison'][6]
 for n in [130,140,141,144,145,148,149,152,155]:
  matching=[r for r in rows if r[10].endswith(':item:'+str(n))]
  assert len(matching)==1 and 'pl' in matching[0][14].split()
 for r in rows:
  if ':item:' in r[10]:
   n=int(r[10].split(':item:')[1].split(':')[0])
   if 156<=n<=219:assert 'verb' in r[14].split()
   if 156<=n<=173:assert 'copula' in r[14].split()
 # Remediation may change grammatical labels/notes, never sampled lexical readings.
 old=list(csv.reader((P/'independent-pass1-frozen/full-staged.csv').open()))
 assert [(r[10],r[2],r[3],r[7]) for r in old]==[(r[10],r[2],r[3],r[7]) for r in rows]

def test_explicit_construction_labels_and_voice_are_not_lost():
 rows,_=source.generate();by={int(r[10].split(':item:')[1].split(':')[0]):r for r in rows if ':item:' in r[10]}
 expectations={169:{'inf'},172:{'1sg','modal'},173:{'1sg','fut'},176:{'inf'},191:{'1sg','pres','progressive'},192:{'1sg','pret','progressive'},193:{'1sg','pret','perfect'},194:{'1sg','modal'},202:{'1sg','pres','pass'},203:{'1sg','pret','pass'},204:{'1sg','fut','pass'}}
 for n,tags in expectations.items():assert tags<=set(by[n][14].split())
 for n in range(168,220):
  if n not in {174,201}:assert 'Source construction label: “'+by[n][3]+'”.' in by[n][6]
 old=list(csv.reader((P/'independent-pass2-frozen/full-staged.csv').open()))
 assert [(r[10],r[2],r[3],r[7]) for r in old]==[(r[10],r[2],r[3],r[7]) for r in rows]
