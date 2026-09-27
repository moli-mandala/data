"""Registered whole Korava source-stage checks, without a database build."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from segments import Tokenizer
DATA = Path(__file__).resolve().parents[1]
P = DATA / 'data/other/forms/raw_data/grierson_korvi_1906'
CSV = DATA / 'data/other/forms/20260925-grierson-korvi.csv'
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(P))

def test_installed_bytes_and_full_regeneration():
    from prepare_full import generate
    rows, audit = generate()
    assert rows == list(csv.reader(CSV.open()))
    assert audit == [json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
    assert CSV.read_bytes() == (P/'proposal.csv').read_bytes()
    assert (P/'audit.jsonl').read_bytes() == (P/'proposal-audit.jsonl').read_bytes()
    assert (DATA/'conversion/grierson-korvi-1906.txt').read_bytes() == (P/'proposal-profile.txt').read_bytes()
    assert len(rows) == 990 and len(audit) == 1237

def test_legacy_keys_and_durable_identity_unchanged():
    previous = json.loads((P/'installation-identity-before-20260926.json').read_text())
    keys = {r[10] for r in csv.reader(CSV.open())}
    assert len(previous['legacy_keys']) == 156
    assert set(previous['legacy_keys']) <= keys
    # The whole identity registry necessarily changes on a full release build.
    # Check this source's immutable keys and their public aliases instead.
    wanted = set(previous['legacy_keys'])
    active = {}
    with (DATA/'data/form-identities.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['Source_Key'] in wanted and row['Status'] == 'active':
                assert row['Source_Key'] not in active
                active[row['Source_Key']] = row['Form_ID']
    assert set(active) == wanted
    assert len(set(active.values())) == len(wanted)
    legacy = {}
    with (DATA/'cldf/form-source-keys.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['Source_Key'] in wanted:
                legacy[row['Legacy_ID']] = row['Source_Key']
    assert set(legacy.values()) == wanted
    aliases = {}
    with (DATA/'cldf/form-id-aliases.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['Legacy_ID'] in legacy:
                aliases[row['Legacy_ID']] = row['Form_ID']
    assert all(aliases.get(old_id, old_id) == active[key] for old_id, key in legacy.items())

def test_actual_registered_parser_round_trip():
    import make_cldf
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name='20260925-grierson-korvi')
    raw = {r[10]:r for r in csv.reader(CSV.open())}
    assert not errors.getvalue(), errors.getvalue()
    assert len(parsed) == stats['converted'] == 990
    for row in parsed:
        r = raw[row.entry_key]
        assert (row.old_form,row.native,row.ipa,row.source,row.tags)==(r[2],r[4],r[5],r[7],r[14])
        assert row.lang == 'Yerukula' and '�' not in row.form

def test_registered_lects_and_source_references():
    import pybtex.database
    from pybtex import PybtexEngine
    dialects = {r[0]:r for r in csv.reader((DATA/'cldf/dialects.csv').open())}
    for key in ['korvi_belgaum_lsi1906','korchari_belgaum_lsi1906','korvi_jamkhandi_lsi1906','korvaru_bijapur_lsi1906']:
        assert dialects[key][2]=='Yerukula' and dialects[key][6:8]==['','']
    bib=pybtex.database.parse_file(str(DATA/'cldf/sources.bib'))
    for key in ['grierson1906korava','grierson_addenda_minora_iv_boundin']:
        assert PybtexEngine().format_from_string(bib.entries[key].to_string('bibtex'),'plain',output_backend='markdown')
    assert 'Korvi 39 installed' not in bib.entries['grierson1906lsi4'].fields['included']

def test_profile_citations_and_graph_are_registered():
    import make_refs,profile_policy
    from unify_cldf import citation_keys
    rows=list(csv.reader(CSV.open()));keys={r[10] for r in rows}
    tokenizer=Tokenizer(str(DATA/'conversion/grierson-korvi-1906.txt'))
    for row in rows:
        assert '�' not in tokenizer(row[2],column='IPA')
        assert set(make_refs.source_ids(row[7]))==citation_keys(row[7])
        assert not row[11] or row[11] in keys
    _,body=profile_policy.read_profile(DATA/'conversion/grierson-korvi-1906.txt')
    rules={r[0]:r[1] if len(r)>1 else '' for r in body}
    assert all(profile_policy.house_output(g,o,rules,'grierson-korvi-1906')==profile_policy.nfc(o) for g,o in rules.items())
