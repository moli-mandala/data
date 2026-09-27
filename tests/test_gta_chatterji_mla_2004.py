"""Full archived Gta scope and malformed-boundary regressions."""
import csv,importlib.util,json,unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
from make_cldf import parse_file
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'data/other/forms/raw_data/gta_chatterji_mla_2004'
INSTALLED=ROOT/'data/other/forms/20260925-donegan-stampe-gta-chatterji.csv'
SPEC=importlib.util.spec_from_file_location('gta_chatterji_mla',PACKAGE/'import_source.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
def data():
 rows,audit=MODULE.prepare();return rows,{r[10].removeprefix('gta-chatterji-mla2004:'):r for r in rows},audit

def test_full_scope_and_stable_identities():
 rows,index,audit=data()
 assert len(audit)==2066 and len(rows)==2259
 assert list(csv.reader(INSTALLED.open()))==rows
 assert [json.loads(l) for l in (PACKAGE/'audit.jsonl').read_text().splitlines()]==audit
 assert len(index)==len(rows)
 assert sum(not r[3] for r in rows)==307
 assert sum(bool(r[11]) for r in rows)==109
 assert all(len(r)==15 and r[0]=='gt' and r[2] and r[7].startswith('DSGT[') for r in rows)
 assert all(unicodedata.is_normalized('NFC',x) and '�' not in x for r in rows for x in r)
 assert all(not r[11] or r[11].removeprefix('gta-chatterji-mla2004:') in index for r in rows)
 assert all(not r[13] or r[13].removeprefix('gta-chatterji-mla2004:') in index for r in rows)

def test_every_witness_and_notation_preserved():
 _,r,a=data()
 assert r['11'][2:4]==['a','negative'] and 'prefix' in r['11'][14]
 assert r['120'][2]=='a-n@G'
 assert r['4351'][2]=='Dia' and r['4351:variant:2'][2]=='nDia'
 assert not r['4351:variant:2'][11]
 assert r['2181'][2]=='b[n]ok-DaiG'
 assert r['341'][2:4]==['aha~','']
 assert r['4991'][3]==''

def test_multisense_and_recovered_malformed_lines():
 _,r,a=data()
 assert r['631'][3]=='numerous, more' and r['631:sense:2'][3]=='very'
 assert r['631:sense:2:variant:3'][2]=='hanToa'
 assert r['6172'][2:4]==['boira-gula','deaf-mute person']
 assert r['11031'][3]=='thin beam' and r['13272'][3]=='thin beam'
 assert r['unnumbered:1'][2:4]==['go-gu','seventeen (lit. ten seven)']
 assert r['unnumbered:2'][2:4]==['jibon-lEe-ke-ne','live']
 assert r['unnumbered:3'][2:4]==['jibon-lEe-ke-ne','live']
 assert r['7772'][2]=='jibon-lEe-ke-ne remoa'
 assert '1112:sense:2' not in r
 assert r['1112'][3]=='to bow (bend)'
 assert r['1112:causative:1'][2:4]==['a-baG-Toe-','to bend']
 assert r['1112:causative:1'][13]=='gta-chatterji-mla2004:1112'
 assert {'verb','intr','caus'} <= set(r['1112:causative:1'][14].split())
 assert r['30'][3]=='to burn' and 'tr' in r['30'][14].split()
 assert r['15050:variant:2'][11]==''

def test_profile_and_scoped_pipeline():
 rows,_,_=data();tokenizer=Tokenizer(str(ROOT/'conversion/gta-chatterji-mla.txt'))
 for r in rows:
  result=tokenizer(r[2],column='IPA').replace(' ','').replace('#',' ')
  assert result==r[2].replace('w','v') and '�' not in result
 errors=[];parsed,stats=parse_file(str(INSTALLED),errors=errors)
 assert not errors and len(parsed)==2259
 assert len({x.entry_key for x in parsed})==2259
 assert any(x.old_form=='b[n]ok-DaiG' for x in parsed)
 assert any(x.old_form=='a-n@G' for x in parsed)
