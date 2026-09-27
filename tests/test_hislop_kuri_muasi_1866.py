"""Full historical Kuri/Muasi source-stage coverage and regression checks."""
import csv,importlib.util,json,sys
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(DATA))
PACKAGE=DATA/'data/other/forms/raw_data/hislop_kuri_muasi_1866'
CSV=DATA/'data/other/forms/20260925-hislop-kuri-muasi.csv'
PROFILE=DATA/'conversion/hislop-kuri-muasi-1866.txt'
spec=importlib.util.spec_from_file_location('hislop',PACKAGE/'import_source.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

def test_complete_accounting_and_keys():
 rows,a=source.build()
 assert len(rows)==326 and len(a)==464
 assert Counter(x['section'] for x in a)=={'vocabulary':359,'comparison':100,'essay':5}
 assert Counter(x['status'] for x in a)=={'selected':295,'source_blank':154,'held':2,'reused_same_source_attestation':13}
 assert len({r[10] for r in rows})==326
 assert all(len(r)==15 and r[0]=='ko' and r[2] and not any(r[n] for n in [1,4,5,8,11,12,13]) for r in rows)
 assert {r[10] for r in rows if r[3]=='basket'}=={'hislop1866papers:kuri:p4:item01:alt1','hislop1866papers:kuri:p4:item01:alt2'}
 assert all(r['note'] and r['raw_record'] for r in a)
 assert sum(len(x['emitted_keys']) for x in a)==326

def test_transcription_repairs_and_form_scope():
 rows,a=source.build();by={r[10]:r for r in rows};key='hislop1866papers:kuri:'
 assert by[key+'p2:item02'][2]=='Jánwar'
 assert by[key+'p3:item07'][2]=='Tēili'
 assert by[key+'p4:item02'][2:4]==['Danyá','be'] and 'verb' in by[key+'p4:item02'][14].split()
 assert by[key+'p20:item04'][2]=='Kisí'
 assert by[key+'p24:item02'][2]=='Ing`' and 'uncertain' in by[key+'p24:item02'][14].split()
 assert 'adj' not in by[key+'p22:item04:alt1'][14].split()
 assert 'adj' in by[key+'p22:item04:alt2'][14].split()
 assert by[key+'p27:item08:alt1'][3]=='plantain'
 assert by[key+'p27:item08:alt2'][3]=='wild plantain'
 assert by[key+'comparison:item07:hislop'][2]=='Minnco'
 assert by[key+'comparison:item16:hislop'][2]=='Gomci'
 assert len([x for x in a if x['status']=='held'])==2
 assert all(x['witness']=='Elliott' and x['review_flags'] for x in a if x['status']=='held')
 assert all('uncertain' in r[14].split() and r[9] for r in rows if ':essay:' in r[10])

def test_reuse_keeps_both_citations_and_collector_witnesses():
 rows,a=source.build();by={r[10]:r for r in rows}
 for x in a:
  if x['reuse_entry_keys']:
   assert x['witness']=='Hislop' and x['section']=='comparison'
   assert len(x['reuse_entry_keys'])==1
   row=by[x['reuse_entry_keys'][0]]
   assert len(row[7].split(';'))==2
   assert row[2]==x['source_forms'][0] and row[3]==x['gloss']
 assert len([r for r in rows if r[3]=='water'])==5 # main + three collectors + essay
 assert all(not r[11] for r in rows) # co-listed synonyms are not inferred sound variants

def test_profiles_tags_dialects_and_bibliography():
 rows,a=source.build();tok=Tokenizer(str(PROFILE))
 for row in rows:
  assert tok(row[2],column='IPA').replace(' ','')==row[2].lower().replace('w','v')
 assert tok('Bhawadi',column='IPA').replace(' ','')=='bhavadi' # no unsupported aspirate phonology
 from tags import GRAMMATICAL_TAGS
 dialects={r['Tag']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
 for row in rows:
  for tag in row[14].split():
   if tag.startswith('dialect:'):
    assert dialects[tag]['Language_ID']=='ko'
    assert not dialects[tag]['Latitude'] and not dialects[tag]['Longitude']
   else:assert tag in GRAMMATICAL_TAGS
 assert (DATA/'cldf/sources.bib').read_text().count('@book{hislop1866papers,')==1


def test_installed_reproducible_and_independent_audit():
 rows,_=source.build()
 assert rows==list(csv.reader(CSV.open()))
 import hashlib
 report=json.loads((PACKAGE/"independent-audit-20260926-pass1.json").read_text())
 assert report["material_errors"]==0 and len(report["sample"])==report["sample_size"]==20
 assert hashlib.sha256(CSV.read_bytes()).hexdigest()==report["hashes"]["staged.csv"]
