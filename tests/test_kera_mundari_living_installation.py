"""Scoped checks of the real registered source; no compiled build."""
import csv
import hashlib
import io
import json
from pathlib import Path

import make_cldf
import make_refs
from pybtex.database import parse_file
from source_meta import SourceMeta

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/other/forms/raw_data/kera_mundari_living_2026'
CANON=ROOT/'data/other/forms/20260927-kera-mundari-living.csv'

def test_registered_parser_all_rows_and_source_graph():
    assert CANON.read_bytes()==(P/'proposal.csv').read_bytes()
    raw={r[10]:r for r in csv.reader(CANON.open())}
    errors=io.StringIO(); parsed,_=make_cldf.parse_file(str(CANON),errors)
    assert not errors.getvalue() and len(parsed)==len(raw)==479
    for row in parsed:
        old=raw[row.entry_key]
        assert row.form==row.old_form==old[2] and row.ipa==row.native==''
        assert set(old[14].split())<=set(row.tags.split())
        assert list(make_refs.source_ids(old[7]))==['living-kera-mundari2026']
        assert not old[11] or old[11] in raw
        assert not old[12] and not old[13]
    assert sum(bool(r[11]) for r in raw.values())==17
    assert raw['living-kera-mundari2026:about:past-keda:reading:1'][14]=='suffix pret'

def test_registered_dialects_reference_profile_and_identity():
    expected=list(csv.DictReader((P/'proposed-dialect.csv').open()))
    actual={r['ID']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    assert all(actual[r['ID']]==r for r in expected)
    assert (ROOT/'conversion/kera-mundari-living-literal.txt').read_bytes()==(P/'proposal-profile.txt').read_bytes()
    assert (P/'installed-audit.jsonl').read_bytes()==(P/'proposal-audit.jsonl').read_bytes()
    references=parse_file(str(ROOT/'cldf/sources.bib')).entries
    assert '479 forms' in references['living-kera-mundari2026'].fields['included']
    assert 'not independently consulted' in references['kobayashi_murmu2008kera'].fields['included']
    meta=SourceMeta()
    rule=meta.transcription_rule('living-kera-mundari2026',CANON,'mu')
    assert rule['convert'] is False and rule['preserve_hyphens'] is True
    assert meta.flag('living-kera-mundari2026','reference','etymology_provenance')=='source'
    receipt=json.loads((P/'installation-report-20260927.json').read_text())
    assert hashlib.sha256((ROOT/'data/form-identities.csv').read_bytes()).hexdigest()==receipt['durable_identity_sha256_unchanged']
