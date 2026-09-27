"""Focused regressions for the full Kiunthali chapter, Bailey 1908."""
import csv, hashlib, importlib.util, io, json, unicodedata
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_kiunthali_1908'
CSV=DATA/'data/other/forms/20260925-bailey-kiunthali.csv'
PROFILE=DATA/'conversion/bailey-kiunthali-1908.txt'
spec=importlib.util.spec_from_file_location('bailey_kiunthali_1908',P/'import_source.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def installed():return list(csv.reader(CSV.open(encoding='utf-8',newline='')))
def audited():return [json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
def row(page,section,item,answer=1):
 key=f'bailey1908kiunthali:p{page}:{section}:item:{item}'
 if answer>1:key+=f':answer{answer}'
 return {r[10]:r for r in installed()}[key]

def test_full_chapter_regeneration():
 rows,audit=source.generate()
 assert rows==installed() and audit==audited()
 assert len(rows)==617 and len(audit)==515
 assert Counter(a['status'] for a in audit)=={'ingested':505,'hold_morphology':1,'exclude_pattern':4,'exclude_other_lect':3,'exclude_other_language':2}
 assert {a['printed_page'] for a in audit}==set(range(11,21))
 assert all(a['scan_page']==a['printed_page']+22 for a in audit)
 assert len({r[10] for r in rows})==len(rows)

def test_legacy_identity_and_corrections():
 old=list(csv.reader((P/'legacy-pilot.csv').open()));new={r[10]:r for r in installed()}
 assert len(old)==13 and {r[10] for r in old}<=new.keys()
 corrections=json.loads((P/'legacy-reading-corrections-20260926.json').read_text())
 assert len(corrections)==5
 for c in corrections:assert new[c['entry_key']][2:4]==[c['after_form'],c['after_gloss']]
 assert row(18,'left',2)[2]=='gihū̃'
 assert row(18,'left',10)[2]=='phaḷ'
 assert row(18,'left',25)[2]=='pāṇī'

def test_paradigms_and_exact_hold():
 assert row(11,'noun-horse',1)[2]=='gōhrā'
 assert row(17,'left',15)[2]=='gōhṛā'
 assert row(11,'noun-father',3)[2]=='bāā khē'
 assert row(11,'noun-father',3,2)[2]=='bā hāgē'
 assert row(12,'noun-cow',3)[2]=='gāūīē'
 held=[a for a in audited() if a['status']=='hold_morphology']
 assert len(held)==1 and held[0]['printed_form_review']=='ĕ ŏ' and not held[0]['entry_keys']
 assert all(not a['entry_keys'] for a in audited() if a['status']!='ingested')

def test_typography_and_source_glosses():
 assert row(17,'left',4)[2]=='be͞uhṇ'
 assert row(17,'left',17,2)[2]=='beuḷd' and 'eu in italics' in row(17,'left',17,2)[6]
 assert row(19,'right',2)[2]=='tsuŋgṇu'
 assert row(14,'preposition-left',5)[2]=='bicc'
 assert row(14,'preposition-left',5,2)[2]=='mānj ṭhē̃'
 assert row(16,'irregular-speak',1)[3]==row(16,'irregular-speak',2)[3]=='say'
 assert row(18,'right',20)[3]=='speak'

def test_whole_expressions_and_alternatives():
 numbered=[a for a in audited() if a['section']=='numbered-specimen']
 assert len(numbered)==22 and all(a['entry_keys'] for a in numbered)
 assert len([a for a in audited() if a['section'] in {'numbered-specimen','verb-note-example','compound-example'}])==36
 assert row(16,'verb-note-example',5)[2]=='tōē̃ nī̃h ēhrū ānthī'
 assert row(16,'verb-note-example',5,2)[2]=='tōē̃ nī̃h ēhrā ānthī'
 assert row(17,'verb-note-example',11)[2]=='ā̃ jāṇu tĕs'
 assert row(17,'verb-note-example',11,2)[2]=='ā̃ jāṇu tĕs khē'
 assert row(19,'numbered-specimen',1)[2].endswith('?')
 assert row(19,'numbered-specimen',1)[6]==''
 assert 'sentential' in row(19,'numbered-specimen',1)[14]

def test_independent_review_pins_final_csv():
 review=json.loads((P/'independent-audit-20260926-pass1.json').read_text())
 final=json.loads((P/'independent-audit-20260926-pass1-addendum.json').read_text())
 assert review['sample_size']==20 and review['error_count']==0
 assert all(e['result']=='pass' for e in review['entries'])
 assert final['sample_material_fields_unchanged']
 assert final['csv_sha256']==hashlib.sha256(CSV.read_bytes()).hexdigest()

def test_literal_profile_and_tags():
 import tags, profile_policy
 t=Tokenizer(str(PROFILE))
 for r in installed():
  assert len(r)==15
  expected=r[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
  assert unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))==expected
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
 assert 'bailey-kiunthali-1908' not in profile_policy.audit({})

def test_metadata_and_scoped_parse():
 import source_meta, make_cldf
 assert source_meta.SourceMeta().transcription('bailey1908kiunthali',CSV,'kiuth')[0]=='bailey-kiunthali-1908'
 assert '@book{bailey1908kiunthali,' in (DATA/'cldf/sources.bib').read_text()
 assert 'kiuth' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
 e=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),e,name='20260925-bailey-kiunthali')
 assert not e.getvalue() and len(parsed)==stats['converted']==617
 raw={r[10]:r for r in installed()}
 assert {r.entry_key for r in parsed}==raw.keys()
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
