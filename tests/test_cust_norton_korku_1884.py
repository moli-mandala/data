"""Complete forward/reverse/numeral/lexical-notes source-stage checks."""
import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/norton_korku_1884'
sys.path.insert(0,str(DATA));sys.path.insert(0,str(PACKAGE))
spec=importlib.util.spec_from_file_location('norton_korku',PACKAGE/'prepare_full_recovery.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
CSV=DATA/'data/other/forms/20260925-cust-norton-korku.csv'
def installed():
 return list(csv.reader(CSV.open()))
def test_complete_inventory_and_reproducibility():
 rows,audit,_=source.build()
 assert rows==installed()
 assert len(audit)==958 and len(rows)==1076
 assert len(rows)==len({r[10] for r in rows})
 assert len(audit)==len({r['entry_key'] for r in audit})
 assert Counter(a['printed_page'] for a in audit if a.get('section')=='Kor–English')=={172:56,173:75,174:78,175:80,176:75,177:63}
 assert sum(a.get('section')=='Grammatical notes' for a in audit)==42
 assert sum(a['status']=='held' for a in audit)==0
 assert sum(a['status'].startswith('excluded') for a in audit)==2
 assert {a['pdf_page']-a['printed_page'] for a in audit}=={19}
def test_fused_and_omitted_heads_and_conflicts():
 rows,audit,_=source.build();bykey={a['entry_key']:a for a in audit}
 base='cust1884korku:kor-english:'
 assert bykey[base+'p174:right:item23']['printed_response']=='jūrī'
 assert bykey[base+'p174:right:item24']['status']=='selected'
 assert bykey[base+'p173:right:item28']['printed_response']=='dotā'
 assert bykey[base+'p177:left:item33']['printed_response']=='topā'
 assert bykey[base+'p177:right:item30']['printed_response']=='yīñ'
 assert bykey[base+'p174:right:item35']['status']=='selected'
 assert any(r[2]=='dobāko' and ' pl' in r[14] and r[13].endswith('item02') for r in rows)
 assert any(r[2]=='korkū' and ' pl' in r[14] and r[13].endswith('item04') for r in rows)
 assert any(r[2]=='tarpaikē' and ' impv' in r[14] for r in rows)
 assert any(r[2]=='tarpaike' and ' pret' in r[14] for r in rows)
 assert any(r[2]=='mahina' and r[3]=='mouth' and 'uncertain' in r[14] for r in rows)
 assert any(r[2]=='mahina' and r[3]=='month' for r in rows)
def test_identity_profile_and_parse():
 rows=installed()
 assert all(len(r)==15 and r[0]=='ko' and source.DIALECT in r[14] for r in rows)
 assert all(r[7].startswith('cust1884korku[p. ') for r in rows)
 assert all(not r[1] and not r[8] and not r[12] for r in rows)
 keys={r[10] for r in rows}
 assert all(not r[13] or r[13] in keys for r in rows)
 old=list(csv.DictReader((PACKAGE/'reviewed_inventory.tsv').open(),delimiter='\t'))
 for item in old:
  if item['status']=='selected':
   prefix=f"cust1884korku:english-kor:p165:{item['column']}:item{int(item['item']):02}"
   assert any(k==prefix or k.startswith(prefix+':alt') for k in keys)
 tokenizer=Tokenizer(str(DATA/'conversion/cust-norton-korku-1884.txt'))
 for r in rows: assert '�' not in tokenizer(r[2],column='IPA')
 import make_cldf
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
 assert not errors.getvalue()
 assert len(parsed)==stats['converted']==len(rows)

def test_forward_pronoun_label_is_not_noun_substring():
 rows,audit,_=source.build()
 prefixes={a['entry_key'] for a in audit if a.get('historical_note',a['note'])=='Direct pronoun'}
 selected=[r for r in rows if any(r[10]==key or r[10].startswith(key+':') for key in prefixes)]
 assert selected
 assert all('pron' in r[14].split() and 'noun' not in r[14].split() for r in selected)
