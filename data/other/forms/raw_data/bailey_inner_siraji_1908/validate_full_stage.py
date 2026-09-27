"""Source-only validation; no database or full CLDF build."""
import csv, importlib.util, io, json, sys, unicodedata
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
DATA=P.parents[4]
sys.path.insert(0,str(DATA))
from segments.tokenizer import Tokenizer
import tags
s=importlib.util.spec_from_file_location('inner_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows,audit=m.generate()
assert rows==list(csv.reader((P/'literal-staged.csv').open()))
assert audit==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
bykey={r[10]:r for r in rows}
legacy=list(csv.reader((P/'legacy-pilot.csv').open()))
assert len(legacy)==88 and {r[10] for r in legacy}<=bykey.keys()
assert len(rows)==500 and len(audit)==419
assert Counter(a['status'] for a in audit)=={'ingested':415,'exclude_pattern':4}
def row(page,section,item,answer=1):
 key=f'bailey1908innersiraji:p{page}:{section}:item:{item}'
 if answer>1:key+=f':answer{answer}'
 return bykey[key]
# Actual old failure classes: missing bottom-of-column cells, transparent suffixes,
# literal breve/underdot/nasal readings, English gloss boundary, and full sentences.
assert all(row(49,'left',i) for i in range(33,43))
assert row(49,'left',6)[2]=='bākrī'
assert row(49,'left',15)[2]=='kukkṛī'
assert row(49,'left',17)[2]=='barĕāḷī'
assert row(49,'left',26)[2]=='kaṇēṭ' and 'lobe of ear?' in row(49,'left',26)[6]
assert row(49,'right',14)[2]=='ghī' and row(49,'right',14,2)[2]=='ghīū'
assert 'bailey1908innersiraji:p49:right:item:14:answer3' not in bykey
assert row(49,'right',30)[2]=='bŏṛau'
assert row(49,'right',12)[2]=='duddh' and 'italicizes u' in row(49,'right',12)[6]
assert row(50,'right',10)[2]=='nī̃ṇā'
assert row(44,'noun-horse',7)[2]=='ghōṛĕā'
assert row(44,'noun-horse',7,2)[2]=='ghōṛĕō'
assert row(46,'auxiliary-present','3sg')[2]=='āsū'
assert row(51,'sentence',8)[2]=='Īmrī piṭṭhī paraundē zīn kŏs̲h̲ā.'
assert row(51,'sentence',8,2)[2]=='Īmrī piṭṭhī uppur zīn kŏs̲h̲ā.'
assert len([a for a in audit if a['section']=='sentence'])==22
assert len([a for a in audit if a['section']=='grammar-expression'])==5
assert all(a['entry_keys'] for a in audit if a['section'] in {'sentence','grammar-expression'})
assert all(not a['entry_keys'] for a in audit if a['status']!='ingested')
assert set(' '.join(r[14] for r in rows).split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
profile=Tokenizer(str(P/'literal-profile-staged.txt'))
for r in rows:
 expected=r[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
 got=unicodedata.normalize('NFC',profile(r[2],column='IPA').replace(' ','').replace('#',' '))
 assert got==expected,(r,got)
import make_cldf
make_cldf.convertors['bailey-inner-siraji-1908']=profile
e=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'literal-staged.csv'),e,name='20260925-bailey-inner-siraji')
assert not e.getvalue(),e.getvalue()
assert len(parsed)==stats['converted']==len(rows)
assert {r.entry_key for r in parsed}==bykey.keys()
assert all(r.old_form==bykey[r.entry_key][2] for r in parsed)
print('500 forms: regeneration, 88 legacy identities, source edge cases, full expressions, profile, tags, and scoped parse pass')
