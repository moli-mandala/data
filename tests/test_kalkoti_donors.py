import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/params/raw_data'

def test_reviewed_donor_regeneration():
    spec = importlib.util.spec_from_file_location('kalkoti_donors', RAW / 'kalkoti_donors.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUTPUT.read_bytes() == mod.render().encode()

def test_homonyms_and_language_mapping():
    a = {r['proposal']: r for r in json.loads((RAW / '20260909-kalkoti-donors-audit.json').read_text())}
    assert len({a[n]['ID'] for n in (376, 377, 378)}) == 3
    assert [a[n]['Gloss'] for n in (376, 377, 378)] == ['gold', 'thousand', 'poison']
    assert a[362]['Language_ID'] == 'Psht'  # not the old mislabeled Arabic compound
    assert a[387]['ID'].endswith('31562')  # room, not camera or married woman
    assert a[391]['Language_ID'] == 'H'  # regional ready, not flying
    assert a[379]['Form'] == 'ustād' and 'ustāz' in a[379]['Original']
    langs = {r['ID'] for r in csv.DictReader((ROOT / 'cldf/languages.csv').open())}
    assert {r['Language_ID'] for r in a.values()} <= langs

def test_compiled_approved_batch():
    p = json.loads((RAW / '20260909-kalkoti-approved-batch17.json').read_text())
    forms = {r['ID']: r for r in csv.DictReader((ROOT / 'cldf/forms.csv').open())}
    edges = {(r['Child_ID'], r['Parent_ID'], r['Kind'], r['Rank'], r['Pos'])
             for r in csv.DictReader((ROOT / 'cldf/edges.csv').open())}
    assert len(p) == 68
    assert len({f for r in p for f in r['formIds']}) == 84
    for r in p:
        parents = r.get('components') or [{'parent': r['parent'], 'pos': ''}]
        if r.get('donor'):
            assert forms[r['parent']]['Status'] == 'entry'
        for f in r['formIds']:
            assert forms[f]['Language_ID'] == 'Kalk'
            assert forms[f]['Status'] == ''
            for par in parents:
                assert (f, par['parent'], r['kind'], '1', str(par['pos'])) in edges

def test_compiled_citations_resolve_without_split_locators():
    p = json.loads((RAW / '20260909-kalkoti-approved-batch17.json').read_text())
    refs = {r['ID'] for r in csv.DictReader((ROOT / 'cldf/references.csv').open())}
    ids = {r['parent'] for r in p if r.get('donor')}
    ids.update(f for r in p for f in r['formIds'])
    for r in csv.DictReader((ROOT / 'cldf/forms.csv').open()):
        if r['ID'] in ids:
            for citation in r['Source'].split(';'):
                assert citation.count('[') == citation.count(']')
                assert citation.split('[', 1)[0] in refs
