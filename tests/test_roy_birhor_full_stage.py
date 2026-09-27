"""Focused integrity checks for Roy's complete staged source, without DB build."""
import csv,importlib.util,unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/roy_birhor_1925'
s=importlib.util.spec_from_file_location('roy_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_complete_accounting_and_reproducibility():
 rows,audit=m.generate()
 assert rows==list(csv.reader((P/'full-preview.csv').open()))
 assert len(rows)==1019 and len(audit)==989
 assert len({r[10] for r in rows})==1019
 assert {r['printed_page'] for r in audit}==set(range(560,592))
 assert sum(r['source_section']=='vocabulary' for r in audit)==875
 assert sum(r['source_section']=='introduction' for r in audit)==114
 assert sum(len(r['reuse_entry_keys']) for r in audit)==9
 assert [r['entry_key'] for r in audit if r['disposition']=='specific_damaged_print_hold']==['roy1925birhor:p574:L14']

def test_legacy_keys_citations_and_comparisons_not_graph_edges():
 rows,_=m.generate();k={r[10]:r for r in rows}
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==46 and {r[10] for r in old}<=k.keys()
 assert all(not r[8] and not r[9] and not r[12] and not r[13] for r in rows)
 assert 'Roy comparison:' in k['roy1925birhor:p589:R19'][6]
 assert 'p. 589' in k['roy1925birhor:p589:R19'][7]
 assert all(p in k['roy1925birhor:p567:R11'][7] for p in ['p. 567','p. 561'])
 assert not k['roy1925birhor:p573:R11:sub01'][6]

def test_specific_damage_uncertainty_and_subentries():
 rows,_=m.generate();k={r[10]:r for r in rows}
 assert 'roy1925birhor:p574:L14' not in k
 assert k['roy1925birhor:p566:prose:01'][2]=='māhā'
 assert k['roy1925birhor:p563:prose:09'][2]=='ḳūndūṛām'
 assert 'uncertain' in k['roy1925birhor:p563:prose:09'][14]
 assert k['roy1925birhor:p584:L05:answer2'][2]=='ūdhrū'
 assert 'uncertain' in k['roy1925birhor:p584:L05:answer2'][14]
 assert k['roy1925birhor:p584:L05:answer2'][11]=='roy1925birhor:p584:L05'
 assert k['roy1925birhor:p573:R11:sub01'][3]=='two days after tomorrow'
 assert k['roy1925birhor:p578:L14:sub01'][2]=='Jid'

def test_literal_profile_coverage():
 rows,_=m.generate();t=Tokenizer(str(P/'full-profile-staged.txt'))
 for r in rows:
  actual=unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==r[2].lower().replace('ng','ŋ').replace('w','v').replace('ṅ','ŋ')
  assert r[2]==unicodedata.normalize('NFC',r[2])

def test_audit_regressions_split_all_head_alternatives():
 rows,_=m.generate();k={r[10]:r for r in rows}
 assert all(',' not in r[2] and ';' not in r[2] for r in rows)
 assert k['roy1925birhor:p579:R19'][2]=='Kumbuṛū'
 assert k['roy1925birhor:p580:L16'][2]=='Lāngṛā'
 assert k['roy1925birhor:p582:R07'][2]=='Nāchā'
 assert k['roy1925birhor:p582:R07:answer2'][2]=='dōriā nāchā'
 assert k['roy1925birhor:p584:R02:answer2'][2]=='Peṭe hūṛū'
 assert k['roy1925birhor:p569:L06:answer3'][2]=='Bir-būrhiā'
 assert k['roy1925birhor:p569:L06:answer3'][11]=='roy1925birhor:p569:L06'

def test_staged_scoped_parser_preserves_editorial_fields():
 import io,make_cldf
 old=make_cldf.convertors.get('roy-birhor')
 try:
  make_cldf.convertors['roy-birhor']=Tokenizer(str(P/'full-profile-staged.txt'))
  errors=io.StringIO()
  parsed,stats=make_cldf.parse_file(str(P/'full-preview.csv'),errors,name='20260925-roy-birhor-p567-p568')
  assert len(parsed)==stats['converted']==1019 and not errors.getvalue()
 finally:
  if old is None:make_cldf.convertors.pop('roy-birhor',None)
  else:make_cldf.convertors['roy-birhor']=old

def test_second_audit_diacritics_and_locative_gloss():
 rows,_=m.generate();k={r[10]:r for r in rows}
 assert k['roy1925birhor:p582:L07'][2]=='Mid-jāng'
 assert 'Itij toṛāng' in k['roy1925birhor:p582:L07'][6]
 assert 'verb' not in k['roy1925birhor:p582:L04'][14].split()

def test_independent_introduction_review_corrections_and_reuse():
 rows,audit=m.generate();k={r[10]:r for r in rows}
 expected={'p562:prose:01':'dūrbal','p562:prose:23':'lūcho','p563:prose:06':'sūpū','p563:prose:20':'āngur lūlūhā'}
 for key,form in expected.items():
  assert k['roy1925birhor:'+key][2]==form
 monkey=next(r for r in audit if r['entry_key']=='roy1925birhor:p564:prose:03')
 assert monkey['reuse_entry_keys']==['roy1925birhor:p574:R03']
 assert k['roy1925birhor:p574:R03'][2]=='gāṛi'
 assert all(page in k['roy1925birhor:p574:R03'][7] for page in ['p. 564','p. 574'])

def test_fourth_audit_comparison_reading():
 rows,_=m.generate();k={r[10]:r for r in rows}
 assert 'Hurumsuku' in k['roy1925birhor:p576:R05'][6]
 assert 'Harumsuku' not in k['roy1925birhor:p576:R05'][6]
 assert not k['roy1925birhor:p576:R05'][8]

def test_full_independent_vocabulary_review_matches_current_inventory():
 import json
 inventory={r['entry_key']:r for r in map(json.loads,(P/'full-reviewed-inventory-staged.jsonl').read_text().splitlines()) if r['source_section']=='vocabulary'}
 seen=set()
 for page in range(567,592):
  report=json.loads((P/f'vocabulary-independent-p{page}-20260926.json').read_text())
  fixes={(f['entry_key'],f['field']):f['source'] for f in report['findings']}
  assert report['units_reviewed']==len(report['coverage'])
  for reviewed in report['coverage']:
   key=reviewed['entry_key'];assert key not in seen;seen.add(key)
   actual=inventory[key]
   for recorded,field in [('form','printed_form'),('gloss','gloss'),('comparison','source_comparison')]:
    expected=fixes.get((key,field),reviewed[recorded])
    assert actual.get(field,'')==expected,(key,field,actual.get(field),expected)
 assert seen==inventory.keys() and len(seen)==875

def test_installed_source_is_the_reviewed_whole_appendix():
 import hashlib,json
 rows,_=m.generate()
 assert rows==list(csv.reader((P.parents[1]/'20260925-roy-birhor-p567-p568.csv').open()))
 assert (P/'full-profile-staged.txt').read_bytes()==(P.parents[4]/'conversion/roy-birhor.txt').read_bytes()
 report=json.loads((P/'independent-full-audit-20260926-pass6.json').read_text())
 assert report['status']=='passed' and report['error_cells']==0 and report['sample_size']==20
 for filename,digest in report['hashes'].items():
  assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
 assert 'import_source_full.py' in (P.parents[1]/'20260925-roy-birhor-p567-p568.yaml').read_text()
