import csv
import importlib.util
import json
import os
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/params/raw_data'
AUDIT = RAW / '20260911-mewari-donors-audit.json'
MANIFEST = ROOT / 'curation/etymology-lab/mewari_dholpura/loan-donors-20260911.json'


def audit():
    return json.loads(AUDIT.read_text())


def test_reproducible_subset_and_source_coverage():
    spec = importlib.util.spec_from_file_location('mewari_donors', RAW / 'mewari_donors.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_bytes() == module.render().encode()
    assert len(audit()) == 22
    assert sum(r['Status'] == 'install' for r in audit()) == 13
    langs = {r['ID'] for r in csv.DictReader((ROOT / 'cldf/languages.csv').open())}
    for row in audit():
        assert row['Language_ID'] in langs
        assert row['Form'] == unicodedata.normalize('NFC', row['Form'])
        assert '\ufffd' not in row['Form']
        assert row['Evidence'] and row['Original'] and row['Transcription']


def test_donor_senses_and_complete_adjectives():
    d = {r['Entry_Key']: r for r in audit()}
    assert d['badan']['Gloss'] == 'body'
    assert d['sabut']['Gloss'] == 'whole; entire'
    assert d['candramas']['Native'] == 'चन्द्रमस्'
    assert d['candramas']['Language_ID'] == 'Indo-Aryan'
    assert d['wazni']['Form'] == 'wazanī'
    assert d['wazn-dar']['Form'] == 'wazn-dār'
    assert d['dasta']['Gloss'].startswith('pestle')
    assert d['bandgobhi']['Native'] == 'बंदगोभी'


def test_registered_identity_survives_reordering_and_gloss_correction():
    from assign_form_ids import assign_ids
    heads = [r for r in audit() if r['Status'] == 'install']
    keys = {r['ID'] for r in heads}
    registry = [r for r in csv.DictReader((ROOT / 'data/form-identities.csv').open()) if r['Legacy_ID'] in keys]
    assert len(registry) == 13 and all(r['Status'] == 'active' for r in registry)
    forms = [dict(ID=r['ID'], Language_ID=r['Language_ID'], Form=r['Form'], Original=r['Form'], Gloss=r['Gloss'], Source=r['Source'], Status='entry') for r in heads]
    expected = {r['ID']: r['Persistent_ID'] for r in heads}
    assert assign_ids(forms, registry)[0] == expected
    corrected = [dict(r, Gloss=r['Gloss'] + '; corrected wording') for r in reversed(forms)]
    assert assign_ids(corrected, registry)[0] == expected


def test_saved_overlay_scope_and_relations():
    m = json.loads(MANIFEST.read_text())
    rows = [r for p in m['analyses'] for r in p['assignments']]
    assert len(rows) == 94
    assert len({r['Form_ID'] for r in rows}) == 88
    existing = {(r['Form_ID'], r['Etymon_ID'], r['Kind'], r['Pos']) for r in csv.DictReader((ROOT / 'data/etymology-assignments.csv').open()) if r['Status'] == 'accepted'}
    assert all((r['Form_ID'], r['Etymon_ID'], r['Kind'], r['Pos']) in existing for r in rows)
    week = next(p for p in m['analyses'] if p['key'] == 'saptaha')
    assert week['parents'] == ['13161'] and week['kind'] == 'borrowed'
    for p in m['analyses']:
        if p['kind'] == 'component':
            for f in p['forms']:
                assert [r['Pos'] for r in p['assignments'] if r['Form_ID'] == f['ID']] == ['1', '2']


@pytest.mark.skipif(not os.environ.get('MEWARI_COMPILED_ROOT'), reason='Requires the completed isolated full data build')
def test_complete_build_contains_donors_and_edges():
    cldf = Path(os.environ['MEWARI_COMPILED_ROOT']) / 'cldf'
    forms = {r['ID']: r for r in csv.DictReader((cldf / 'forms.csv').open())}
    edges = {(r['Child_ID'], r['Parent_ID'], r['Kind'], r['Pos']) for r in csv.DictReader((cldf / 'edges.csv').open()) if r['Rank'] == '1'}
    refs = {r['ID'] for r in csv.DictReader((cldf / 'references.csv').open())}
    for r in audit():
        f = forms[r['Persistent_ID']]
        assert f['Status'] == 'entry' and f['Language_ID'] == r['Language_ID']
        if r['Status'] == 'install':
            assert f['Form'] == r['Form'] and f['Gloss'] == r['Gloss']
        assert all(c.split('[', 1)[0] in refs for c in f['Source'].split(';'))
    m = json.loads(MANIFEST.read_text())
    for p in m['analyses']:
        for a in p['assignments']:
            assert forms[a['Form_ID']]['Status'] == ''
            assert (a['Form_ID'], a['Etymon_ID'], a['Kind'], a['Pos']) in edges
