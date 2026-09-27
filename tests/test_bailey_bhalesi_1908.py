"""Focused checks for the installed complete Bhalesi source stage."""
import csv,hashlib,importlib.util,io,json
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_bhalesi_1908'
CSV=DATA/'data/other/forms/20260925-bailey-bhalesi.csv'
PROFILE=DATA/'conversion/bailey-bhalesi-1908.txt'
spec=importlib.util.spec_from_file_location('bhalesi_installed',P/'import_source.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_exact_audited_installation():
    rows,audit=module.generate()
    assert rows==list(csv.reader(CSV.open())) and len(rows)==435
    assert audit==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()] and len(audit)==436
    assert CSV.read_bytes()==(P/'proposal.csv').read_bytes()
    assert PROFILE.read_bytes()==(P/'proposal-profile.txt').read_bytes()
    report=json.loads((P/'independent-full-audit-20260926-pass3.json').read_text())
    assert report['status']=='passed' and report['material_errors']==0 and report['sample_size']==20
    for name,digest in report['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest

def test_registered_metadata_and_profile():
    import source_meta,profile_policy
    meta=source_meta.SourceMeta()
    assert meta.transcription('bailey1908bhalesi',CSV,'bhal')[0]=='bailey-bhalesi-1908'
    assert meta.flag('bailey1908bhalesi','identity','dedupe_by_entry_key') is True
    assert meta.flag('bailey1908bhalesi','forms','split_alternates') is False
    assert 'bailey-bhalesi-1908' not in profile_policy.audit({})
    import pybtex.database
    entry=pybtex.database.parse_file(str(DATA/'cldf/sources.bib')).entries['bailey1908bhalesi']
    assert '435' in entry.fields['included'] and 'full-reviewed.jsonl' in entry.fields['provenance']
    assert meta.source('bailey1908bhalesi')['reference']['ocr'] is False

def test_actual_generic_parser_roundtrip():
    import make_cldf
    rows={r[10]:r for r in csv.reader(CSV.open())}
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
    assert not errors.getvalue(),errors.getvalue()
    assert len(parsed)==stats['converted']==435
    tokenizer=Tokenizer(str(PROFILE))
    for record in parsed:
        row=rows[record.entry_key]
        assert (record.old_form,record.native,record.ipa,record.source)==(row[2],row[4],row[5],row[7])
        assert record.form==tokenizer(row[2],column='IPA').replace(' ','').replace('#',' ')
