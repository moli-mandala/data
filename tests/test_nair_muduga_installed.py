"""Actual canonical source-stage contract; no compiled dataset or full build."""
import csv
import importlib.util
import io
import json
import tempfile
import unicodedata as ud
from pathlib import Path

from pybtex.database import parse_file
import make_cldf
import source_meta

DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/nair_muduga_2013'
STEM='20260927-nair-muduga'
CSV=DATA/f'data/other/forms/{STEM}.csv'
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')

def rows():return list(csv.reader(CSV.open(newline='')))

def test_registered_canonical_parser_all_fifteen_fields_nfc_nfd_and_keys():
    assert CSV.read_bytes()==(PACKAGE/'proposed.csv').read_bytes()
    raw=rows();meta=source_meta.load()
    assert len(raw)==204 and sum(r[0]=='Muduga' for r in raw)==183
    errors=io.StringIO();canonical,stats=make_cldf.parse_file(str(CSV),errors,name=STEM)
    assert not errors.getvalue() and stats=={'converted':0,'for_conversion':0}
    assert len(canonical)==204
    for parsed,wanted in zip(canonical,raw):
        assert [getattr(parsed,x) for x in FIELDS]==wanted,wanted[10]
        assert parsed.form==wanted[2]
    with tempfile.TemporaryDirectory(dir=PACKAGE/'source-stage') as tmp:
        for norm in ('NFC','NFD'):
            wanted=[[ud.normalize(norm,x) for x in r] for r in raw]
            path=Path(tmp)/f'{norm}.csv'
            with path.open('w',newline='') as f:csv.writer(f).writerows(wanted)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
            assert not errors.getvalue() and stats=={'converted':0,'for_conversion':0}
            assert len(parsed)==len({make_cldf.source_entry_dedupe_key(p,meta) for p in parsed})==204
            for p,r in zip(parsed,wanted):
                assert [getattr(p,x) for x in FIELDS]==r,r[10]
                assert p.form==r[2]
    audit=[json.loads(x) for x in (PACKAGE/'installed-audit.jsonl').read_text().splitlines()]
    assert len(audit)==128 and sum(len(u['physical_witnesses']) for u in audit)==229
    assert all(u['accepted_for_source_stage'] for u in audit)
    assert sum(not w['entry_keys'] for u in audit for w in u['physical_witnesses'])==25
    assert {r[10] for r in raw}=={k for u in audit for k in u['entry_keys']}

def test_actual_source_meta_both_references_and_registered_lect():
    meta=source_meta.load()
    for key in ('nair2013muduga','rajendran1986muduga'):
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013muduga' else 'none')
    assert meta.flag('nair2013muduga','identity','dedupe_by_entry_key') is True
    for lang in ('Muduga','Malayalam'):
        assert meta.transcription('nair2013muduga',str(CSV),lang)==(None,False)
    assert source_meta.importer_commands(meta,STEM)==[['data/other/forms/raw_data/nair_muduga_2013/import_source.py','--install']]
    bib=parse_file(DATA/'cldf/sources.bib')
    assert {'nair2013muduga','rajendran1986muduga'}<=bib.entries.keys()
    assert 'installed-audit.jsonl' in bib.entries['nair2013muduga'].fields['provenance']
    d=[r for r in csv.DictReader((DATA/'cldf/dialects.csv').open()) if r['ID']=='nair2013_muduga_attapady']
    assert len(d)==1 and d[0]['Language_ID']=='Muduga'
    assert d[0]['Glottocode']==d[0]['Latitude']==d[0]['Longitude']==d[0]['Quality']==''
    assert all(d[0]['Tag'] in r[14].split() for r in rows() if r[0]=='Muduga')
    langs={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    assert {r[0] for r in rows()}<=langs.keys() and langs['Muduga']['Glottocode']=='mudu1239'

def test_guarded_repeat_noop_identity_legacy_and_all_approval_pins():
    spec=importlib.util.spec_from_file_location('muduga_installed',PACKAGE/'import_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    approval=m.validate_inputs()
    receipt=json.loads(m.RECEIPT.read_text())
    assert m.file_hash(m.IDENTITY)==receipt['durable_identity_sha256']==approval['durable_identity_sha256']
    before={role:m.file_hash(path) for role,path in m.TARGETS.items()}
    result=m.main(True)
    assert result['canonical_mutations'] is False and result['status'].startswith('already installed')
    assert before=={role:m.file_hash(path) for role,path in m.TARGETS.items()}
