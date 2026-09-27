"""Whole Suketi source-stage gates; no database build."""
import csv,importlib.util,io,json,unicodedata
from collections import Counter
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1];P=DATA/'data/other/forms/raw_data/grierson_suketi_1916'
spec=importlib.util.spec_from_file_location('suketi_full_stage',P/'prepare_full.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def rows():return list(csv.reader((P/'proposal.csv').open()))
def test_full_inventory_and_legacy_keys():
 r,a=source.generate();assert r==rows()
 assert len(r)==353 and len(a)==440
 assert Counter(x['status'] for x in a)=={'ingested':341,'source-blank':4,'other-lect-control':16,'source-scope-ambiguous':79}
 assert len({x[10] for x in r})==353
 assert {k for x in a for k in x['entry_keys']}=={x[10] for x in r}
 assert all(x['review']=='visual-second-pass' for x in a)
 legacy=list(csv.reader((DATA/'data/other/forms/20260925-grierson-suketi.csv').open()))
 assert len({x[10] for x in legacy}&{x[10] for x in r})>=55
 assert {x['prompt'] for x in a if 'prompt' in x}==set(range(1,242))
def test_attribution_and_fine_typography():
 r={x[10]:x for x in rows()};a=source.generate()[1]
 assert all(not x['entry_keys'] for x in a if x['status'] in ['source-scope-ambiguous','other-lect-control'])
 assert r['grierson1916suketi:prompt:50:oblique'][2]=='bhaiṇā'
 assert 'obl' in r['grierson1916suketi:prompt:50:oblique'][14].split()
 for n,form in [(5,'Pañj'),(36,'Mūhā̃'),(53,'Lāṛī'),(60,'Parmēśar'),(72,'Kukaṛ'),(76,'Chiṛū')]:assert r[f'grierson1916suketi:prompt:{n}'][2]==form
 uncertain=r['grierson1916suketi:p758:line:6:cell:3'];assert uncertain[2]=='mukvā' and 'uncertain' in uncertain[14].split()
 assert 'damaged y' in uncertain[6]
 assert 'kōrṛē' in r['grierson1916suketi:prompt:228'][2]
 assert 'rahā̃' in r['grierson1916suketi:prompt:233'][2]
def test_structured_prompt_grammar():
 r={x[10]:x for x in rows()}
 for n,required in [(15,{'pron','1sg','gen'}),(113,{'noun','sg','abl'}),(133,{'adj','degree'}),(156,{'verb','copula','1sg','pres'}),(191,{'1sg','progressive','pres'}),(194,{'subjunctive'}),(203,{'pass','pret'}),(231,{'sentential'})]:assert required<=set(r[f'grierson1916suketi:prompt:{n}'][14].split())
 assert 'subjunctive' in r['grierson1916suketi:p757:verb:1'][14].split()
 assert not any('subj' in x[14].split() for x in r.values())
def test_profile_coverage_and_tags():
 import tags
 t=Tokenizer(str(P/'proposal-profile.txt'))
 for r in rows():
  assert len(r)==15 and not r[4]
  expected=unicodedata.normalize('NFC',r[2]).lower().replace('w','v').replace('ṅ','ŋ')
  assert t(r[2],column='IPA').replace(' ','').replace('#',' ')==expected
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
  assert r[11]==r[12]==r[13]==''
def test_scoped_parser(monkeypatch):
 import make_cldf
 monkeypatch.setitem(make_cldf.convertors,'grierson-suketi-1916',Tokenizer(str(P/'proposal-profile.txt')))
 error=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'proposal.csv'),error,name='20260925-grierson-suketi')
 assert not error.getvalue(),error.getvalue()
 assert len(parsed)==stats['converted']==353
 assert {x.entry_key for x in parsed}=={r[10] for r in rows()}
