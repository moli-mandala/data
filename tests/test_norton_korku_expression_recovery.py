"""Recovery of all printed expressions, with original dictionary identities retained."""
import csv,io,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/norton_korku_1884'
DATA=P.parents[4]
def inventory():
 return list(csv.reader((P/'expression-proposal.csv').open())),[json.loads(x)for x in(P/'expression-proposal-audit.jsonl').read_text().splitlines()]
def test_all_complete_expressions_and_controls_accounted():
 rows,audit=inventory();rr={r[10]:r for r in rows}
 assert len(rows)==1046 and len(audit)==958
 assert len(rr)==1046 and len({a['entry_key']for a in audit})==958
 assert not any(a['status']=='excluded_sentence'for a in audit)
 for n in range(15,29):
  row=rr[f'cust1884korku:notes:p178:item{n:02}'];assert ' 'in row[2] and row[3]
 assert sum(a['status']=='excluded_control'for a in audit)==2
 assert sum(a['status']=='audit_only_metalinguistic'for a in audit)==1
 assert sum(a['status']=='held'for a in audit)==9
 assert rr['cust1884korku:notes:p178:item19'][2]=='gaoen gel ūrāko tīkyē'
 assert rr['cust1884korku:notes:p178:item25'][2]=='in hegibā'
 assert rr['cust1884korku:notes:p178:item21'][2].endswith('yī,en')
 assert rr['cust1884korku:notes:p178:item01:alt1'][2]=='yi,en'
 assert rr['cust1884korku:notes:p178:item01:alt2'][2]=='hē,en'
 assert all(len(r)==15 for r in rows)
def test_legacy_dictionary_rows_unchanged_and_grammar_scoped():
 rows,_=inventory();rr={r[10]:r for r in rows}
 old=list(csv.reader((P/'canonical-before-full-recovery.csv').open()))
 assert {r[10]for r in old}<={r[10]for r in rows}
 assert all(rr[r[10]]==r for r in old if ':notes:'not in r[10])
 for n in [7,8]:
  selected=[r for r in rows if r[10].startswith(f'cust1884korku:notes:p179:item{n:02}')]
  assert selected and all('loanword'in r[14].split()for r in selected)
def test_actual_profile_parser_and_whole_phrase_boundaries():
 import make_cldf
 from segments import Tokenizer
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 rows,_=inventory();tokenizer=Tokenizer(str(DATA/'conversion/cust-norton-korku-1884.txt'))
 for r in rows:
  assert '�'not in tokenizer(r[2],column='IPA')
  assert set(r[14].split())-{t for t in r[14].split()if t.startswith('dialect:')}<=GRAMMATICAL_TAGS|GENDER_TAGS
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'expression-proposal.csv'),errors,name='20260925-cust-norton-korku')
 assert len(parsed)==stats['converted']==1046 and not errors.getvalue()
 assert {r.entry_key:r.old_form for r in parsed}=={r[10]:r[2]for r in rows}
 selected=next(r for r in parsed if r.entry_key=='cust1884korku:notes:p178:item19')
 assert selected.form=='gaoen gel ūrāko tīkyē'
