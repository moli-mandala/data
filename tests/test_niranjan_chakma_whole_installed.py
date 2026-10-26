"""Actual whole-book source-stage contracts; no database build."""
import csv,io,json,unicodedata
from pathlib import Path
import make_cldf,source_meta
D=Path(__file__).resolve().parents[1]
P=D/'data/other/forms/raw_data/niranjan_chakma_2010'
CSV=D/'data/other/forms/20260921-niranjan-chakma.csv'

def test_installed_reviewed_bytes_keys_and_audit_accounting():
 assert CSV.read_bytes()==(P/'whole-export-r2/whole-source-export-preview.csv').read_bytes()
 rows=list(csv.reader(CSV.open()));old=list(csv.reader((P/'whole-installation/legacy-742.csv').open()))
 assert len(rows)==3662 and len({r[10] for r in rows})==3662
 assert [r[10] for r in rows[:742]]==[r[10] for r in old]
 assert (P/'whole-installed-audit.jsonl').read_bytes()==(P/'whole-export-r2/whole-source-export-audit.jsonl').read_bytes()
 audit=[json.loads(x) for x in (P/'whole-installed-audit.jsonl').read_text().splitlines()]
 assert len(audit)==5232 and {k for a in audit for k in a['emitted_entry_keys']}=={r[10] for r in rows}
 assert all(unicodedata.is_normalized('NFC',r[2]) and '\ufffd' not in r[2] for r in rows)

def test_actual_canonical_parser_all_fields_and_no_unwanted_splits():
 rows=list(csv.reader(CSV.open()));errors=io.StringIO()
 parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
 assert not errors.getvalue() and len(parsed)==len(rows) and stats=={'converted':0,'for_conversion':0}
 attrs=['lang','param','form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags']
 for actual,expected in zip(parsed,rows):
  assert actual.old_form==expected[2]
  assert [getattr(actual,k) for k in attrs]==expected

def test_canonical_metadata_route_and_language_registration():
 meta=source_meta.load()
 assert source_meta.importer_commands(meta,CSV.stem)==[['data/other/forms/raw_data/niranjan_chakma_2010/whole-installation/install_whole_source.py','--install']]
 assert meta.file_defaults(CSV)['transcription']==[{'convert':False,'preserve_hyphens':True}]
 assert meta.flag('niranjan2010chakma','reference','ocr') is True
 assert meta.flag('niranjan2010chakma','reference','etymology_provenance')=='source'
 assert meta.flag('niranjan2010chakma','identity','dedupe_by_entry_key') is True
 assert meta.flag('niranjan2010chakma','forms','split_alternates') is False
 rows=list(csv.reader(CSV.open()));registry={r['ID']:r for r in csv.DictReader((D/'cldf/languages.csv').open())}
 assert {r[0] for r in rows}<=set(registry)
 assert sum(r[0]=='Tanchangya' for r in rows)==3 and sum(r[0]=='Chenchu' for r in rows)==4
 assert all(not r[i] for r in rows for i in [1,8,9,11,12,13])
