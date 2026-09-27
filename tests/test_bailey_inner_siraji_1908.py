"""Full chapter source regressions; no full data build."""
import csv,hashlib,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_inner_siraji_1908'
CSV=DATA/'data/other/forms/20260925-bailey-inner-siraji.csv'
PROFILE=DATA/'conversion/bailey-inner-siraji-1908.txt'
s=importlib.util.spec_from_file_location('inner_siraji',P/'import_source.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def installed():return list(csv.reader(CSV.open()))
def audited():return [json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
def row(page,section,item,answer=1):
 key=f'bailey1908innersiraji:p{page}:{section}:item:{item}'+(f':answer{answer}' if answer>1 else '')
 return {r[10]:r for r in installed()}[key]
def test_complete_chapter_regeneration():
 rows,audit=m.generate()
 assert rows==installed() and audit==audited()
 assert len(rows)==500 and len(audit)==419
 assert len({r[10] for r in rows})==500 and len({a['source_cell_key'] for a in audit})==419
 assert Counter(a['status'] for a in audit)=={'ingested':415,'exclude_pattern':4}
 assert {a['printed_page'] for a in audit}==set(range(44,52))
 assert all(a['scan_page']==a['printed_page']+22 for a in audit)
 assert sum(a['section']=='lexical' for a in audit)==119
 assert sum(a['section']=='cardinal' for a in audit)==47
 assert sum(a['section']=='ordinal' for a in audit)==7

def test_old_keys_and_omitted_column_bottom():
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==88 and {r[10] for r in old}<={r[10] for r in installed()}
 assert all(row(49,'left',i) for i in range(33,43))
 assert row(49,'left',33)[2:4]==['katāb','book']
 assert row(49,'left',42)[2:4]==['rōṭṭī','bread']

def test_recovered_suffixes_and_typography():
 assert row(49,'left',6)[2]=='bākrī'
 assert row(49,'left',15)[2]=='kukkṛī'
 assert row(49,'left',17)[2]=='barĕāḷī'
 assert row(49,'left',26)[2]=='kaṇēṭ' and 'lobe of ear?' in row(49,'left',26)[6]
 assert row(49,'right',30)[2]=='bŏṛau'
 assert row(50,'right',10)[2]=='nī̃ṇā'
 assert 'italicizes u' in row(49,'right',12)[6]
 assert 'small and raised' in row(48,'left',6)[6]

def test_gloss_boundary_and_source_inconsistencies():
 assert row(49,'right',14)[2]=='ghī' and row(49,'right',14,2)[2]=='ghīū'
 assert not any(r[10].endswith('p49:right:item:14:answer3') for r in installed())
 assert row(50,'right',6)[2]=='rauhṇa' and row(47,'remain-head',1)[2]=='rauhṇā'
 assert row(50,'cardinal:left',2)[2]=='dūi'
 assert row(50,'cardinal:right',45)[2]=='dūī s̲h̲au'

def test_grammar_and_whole_specimens():
 assert row(44,'noun-horse',7)[2]=='ghōṛĕā' and row(44,'noun-horse',7,2)[2]=='ghōṛĕō'
 assert 'voc' in row(44,'noun-horse',7)[14].split()
 assert row(46,'auxiliary-present','3sg')[2]=='āsū'
 assert {'verb','auxiliary','pres','3sg'}<=set(row(46,'auxiliary-present','3sg')[14].split())
 assert row(51,'sentence',8)[2]=='Īmrī piṭṭhī paraundē zīn kŏs̲h̲ā.'
 assert row(51,'sentence',8,2)[2]=='Īmrī piṭṭhī uppur zīn kŏs̲h̲ā.'
 a=audited()
 assert sum(x['section']=='sentence' for x in a)==22
 assert sum(x['section']=='grammar-expression' for x in a)==5
 assert all(x['entry_keys'] for x in a if x['section'] in {'sentence','grammar-expression'})
 assert all(not x['entry_keys'] for x in a if x['status']!='ingested')

def test_independent_audit_pins_installed_rows():
 r=json.loads((P/'independent-audit-20260926-pass1.json').read_text())
 assert r['status']=='passed' and r['material_errors']==0
 assert r['csv_sha256']==hashlib.sha256(CSV.read_bytes()).hexdigest()
 assert len(r['selection'])==20
 bykey={x[10]:x for x in installed()}
 assert all(x['result']=='pass' and bykey[x['csv_row'][10]]==x['csv_row'] for x in r['selection'])

def test_profile_tags_metadata_and_scoped_parse():
 import make_cldf,profile_policy,source_meta,tags
 rows=installed();t=Tokenizer(str(PROFILE))
 assert set(' '.join(r[14] for r in rows).split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
 for r in rows:
  assert len(r)==15 and r[0]=='insir' and all(not r[i] for i in (1,4,5,8,9,11,12,13))
  got=unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))
  assert got==r[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
 assert source_meta.SourceMeta().transcription('bailey1908innersiraji',CSV,'insir')[0]=='bailey-inner-siraji-1908'
 assert 'bailey-inner-siraji-1908' not in profile_policy.audit({})
 assert '@book{bailey1908innersiraji,' in (DATA/'cldf/sources.bib').read_text()
 assert 'insir' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
 e=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),e,name='20260925-bailey-inner-siraji')
 assert not e.getvalue() and len(parsed)==stats['converted']==500
 bykey={r[10]:r for r in rows}
 assert {r.entry_key for r in parsed}==bykey.keys()
 assert all(r.old_form==bykey[r.entry_key][2] for r in parsed)
