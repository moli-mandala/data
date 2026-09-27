"""Small source-only checks; does not build CLDF or a database."""
import csv
import importlib.util
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[4]
sys.path.insert(0,str(DATA))
from segments.tokenizer import Tokenizer
import tags
s=importlib.util.spec_from_file_location('kiunthali_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows,audit=m.generate(P/'chapter-transcription-work.tsv')
assert rows==list(csv.reader((P/'literal-staged.csv').open()))
assert audit==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
bykey={r[10]:r for r in rows}
legacy=list(csv.reader((P/'legacy-pilot.csv').open()))
assert len(legacy)==13 and {r[10] for r in legacy}<=bykey.keys()
assert len(rows)==617 and len(audit)==515
assert Counter(a['status'] for a in audit)=={'ingested':505,'hold_morphology':1,'exclude_pattern':4,'exclude_other_lect':3,'exclude_other_language':2}
def row(page,section,item,answer=1):
 key=f'bailey1908kiunthali:p{page}:{section}:item:{item}'
 if answer>1:key+=f':answer{answer}'
 return bykey[key]
# Meaningful editorial failure classes: real source stems, literal inconsistencies,
# source style spans, whole translated expressions, and non-lexical exclusions.
assert row(11,'noun-horse',1)[2]=='gōhrā'
assert row(17,'left',15)[2]=='gōhṛā'
assert row(11,'noun-father',3)[2]=='bāā khē'
assert row(11,'noun-father',3,2)[2]=='bā hāgē'
assert row(12,'noun-cow',3)[2]=='gāūīē'
assert row(17,'left',17,2)[2]=='beuḷd' and 'eu in italics' in row(17,'left',17,2)[6]
assert row(17,'left',4)[2]=='be͞uhṇ'
assert row(18,'left',10)[2]=='phaḷ'
assert row(19,'right',2)[2]=='tsuŋgṇu'
assert row(16,'verb-note-example',5)[2]=='tōē̃ nī̃h ēhrū ānthī'
assert row(16,'verb-note-example',5,2)[2]=='tōē̃ nī̃h ēhrā ānthī'
assert row(17,'verb-note-example',11)[2]=='ā̃ jāṇu tĕs'
assert row(17,'verb-note-example',11,2)[2]=='ā̃ jāṇu tĕs khē'
assert len([a for a in audit if a['section']=='numbered-specimen'])==22
assert all(a['entry_keys'] for a in audit if a['section']=='numbered-specimen')
assert [a['printed_form_review'] for a in audit if a['status']=='hold_morphology']==['ĕ ŏ']
assert all(not a['entry_keys'] for a in audit if a['status']!='ingested')
assert set(' '.join(r[14] for r in rows).split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
profile=Tokenizer(str(P/'literal-profile-staged.txt'))
for r in rows:
 expected=r[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
 got=unicodedata.normalize('NFC',profile(r[2],column='IPA').replace(' ','').replace('#',' '))
 assert got==expected,(r,got)
import make_cldf
make_cldf.convertors['bailey-kiunthali-1908']=profile
e=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'literal-staged.csv'),e,name='20260925-bailey-kiunthali')
assert not e.getvalue()
assert len(parsed)==stats['converted']==len(rows)
assert {r.entry_key for r in parsed}==bykey.keys()
assert all(r.old_form==bykey[r.entry_key][2] for r in parsed)
print('617 forms: regeneration, legacy identities, source edge classes, whole-expression scope, literal profile, registered tags, and focused parse all pass')
