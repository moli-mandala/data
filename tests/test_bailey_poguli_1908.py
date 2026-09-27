"""Installed complete Poguli chapter checks; no database generation."""
import csv, importlib.util, io, json, unicodedata
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_poguli_1908'
s=importlib.util.spec_from_file_location('poguli_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((DATA/'data/other/forms/20260925-bailey-poguli.csv').open()))
def row(page,section,item,answer=1):
 k=f'bailey1908poguli:p{page}:{section}:item:{item}'+(f':answer:{answer}' if answer>1 else '')
 return {r[10]:r for r in staged()}[k]
def test_full_scope_and_determinism():
 r,a=m.generate();assert r==staged()
 assert a==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
 assert len(a)==400 and len(r)==460 and len({x[10] for x in r})==460
 assert {x['printed_page'] for x in a}==set(range(51,61))
 assert Counter(x['status'] for x in a)=={'ingested':393,'exclude_pattern':4,'exclude_other_language':1,'exclude_rejected_example':1,'exclude_blank':1}
 counts=Counter(x['section'] for x in a)
 assert [counts[s] for s in ['glossary','specimens','prodigal-son','extracts']]==[100,22,29,11]
 assert all(x['scan_page']==x['printed_page']+224 for x in a)
def test_old_keys_and_recovered_glossary():
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==94 and {r[10] for r in old}<={r[10] for r in staged()}
 for p,c,i,f in [(58,'left',2,'dīh'),(58,'left',30,'yĕī'),(58,'right',42,'K̲h̲udā'),(58,'right',67,'dhaũtulnu'),(59,'left',84,'ghōṛĭ'),(59,'right',99,'harnī')]: assert row(p,c,i)[2]==f
 assert 'bailey1908poguli:p59:left:item:79' not in {r[10] for r in staged()}
def test_literal_variants_and_grammar():
 for p,s,i,f in [(58,'right',51,'gāū̃'),(59,'right',88,'gāŭ'),(54,'pronominal-example',3,'jün'),(54,'pronominal-example',5,'küñ'),(53,'come',1,'yiun'),(53,'come-passive-discussion',1,'yīun'),(53,'give-aor-fut',5,'dēōuth')]:assert row(p,s,i)[2]==f
 assert row(52,'possessive-inflection','tyĕs',3)[2]=='tyĕsau'
 assert {'verb','pret','1sg'}<=set(row(53,'aux-past',1)[14].split())
 assert {'pron','poss','f','pl'}<=set(row(52,'possessive-inflection','tyĕs',4)[14].split())
 assert 'caus' in row(54,'causative-eat',2)[14].split()
def test_whole_expressions():
 assert 'multiword-expression' in row(55,'prodigal-son',21)[14]
 assert 'hnntün' in row(55,'prodigal-son',21)[2]
 assert row(60,'specimens',22)[2]=='gāma sanni dukāndāras laba'
 assert 'source interlinear English:' in row(54,'prodigal-son',1)[6]
def test_literal_profile_and_scoped_parse():
 import tags,make_cldf
 r=staged();t=Tokenizer(str(DATA/'conversion/bailey-poguli-1908.txt'))
 for x in r:
  assert len(x)==15 and x[0]=='pog'
  assert set(x[14].split())<=(tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS|{m.LECT_TAG})
  actual=unicodedata.normalize('NFC',t(x[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==x[2].replace('w','v').replace('ṅ','ŋ')
 old=make_cldf.convertors.get('bailey-poguli-1908')
 try:
  make_cldf.convertors['bailey-poguli-1908']=t
  e=io.StringIO();out,stats=make_cldf.parse_file(str(P/'full-staged.csv'),e,name='20260925-bailey-poguli')
  assert not e.getvalue() and len(out)==stats['converted']==460
 finally:
  if old is None:make_cldf.convertors.pop('bailey-poguli-1908',None)
  else:make_cldf.convertors['bailey-poguli-1908']=old

def test_independent_audit_and_registry():
 import hashlib,source_meta,profile_policy
 report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
 canonical=DATA/'data/other/forms/20260925-bailey-poguli.csv'
 assert report['status']=='passed'
 assert hashlib.sha256(canonical.read_bytes()).hexdigest()==report['csv_sha256']
 assert source_meta.SourceMeta().transcription(m.SOURCE,canonical,'pog')[0]=='bailey-poguli-1908'
 assert 'bailey-poguli-1908' not in profile_policy.audit({})
 d=next(r for r in csv.DictReader((DATA/'cldf/dialects.csv').open()) if r['ID']=='bailey1908-poguli')
 assert d['Language_ID']=='pog' and d['Tag']==m.LECT_TAG
 assert '@book{bailey1908poguli,' in (DATA/'cldf/sources.bib').read_text()
