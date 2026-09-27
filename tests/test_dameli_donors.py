import csv
import importlib.util
import json
import os
import io
import sys
import pytest
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / 'data/other/params/raw_data'

def test_regeneration_and_rich_width():
    spec = importlib.util.spec_from_file_location('dameli_donors', RAW / 'dameli_donors.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.OUTPUT.read_bytes() == m.render().encode()
    assert m.FORMS.read_bytes() == m.render_forms().encode()
    records = list(csv.reader(m.FORMS.open()))
    assert len(records) == 33
    assert all(len(r) == 15 and r[1] and r[10] for r in records)
    assert len({r[10] for r in records}) == 33

def test_source_and_transcription_coverage():
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    a = json.loads((RAW/'20260910-dameli-donors-audit.json').read_text())
    langs = {r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    assert {r['Language_ID'] for r in a} <= langs
    for r in a:
        assert r['Form'] == unicodedata.normalize('NFC', r['Form'])
        assert '\ufffd' not in r['Form']
        assert r['Evidence'] and '[' in r['Source']
        assert all(t in GRAMMATICAL_TAGS | GENDER_TAGS or t.startswith('dialect:') for t in r['Tags'].split())

def test_homonyms_and_donor_languages():
    a = {r['ID']:r for r in json.loads((RAW/'20260910-dameli-donors-audit.json').read_text())}
    assert a['loan-dameli-45-1']['Source'].startswith('oped2026[entry 32639,')
    assert a['loan-dameli-45-1']['Gloss'] == 'pink, rose (colour)'
    assert a['loan-dameli-18-1']['Source'].startswith('oped2026[entry 9569,')
    assert a['loan-dameli-19-8']['Language_ID'] == 'Psht'
    assert a['loan-dameli-58-2']['Language_ID'] == 'H'
    assert {'adv', 'adj'} <= set(a['loan-dameli-32-1']['Tags'].split())
    assert {'adj', 'noun', 'm'} <= set(a['loan-dameli-20-2']['Tags'].split())

def test_cited_source_does_not_reconvert_audited_donors():
    import make_cldf
    errors = io.StringIO()
    rows, _ = make_cldf.parse_file(
        'data/other/forms/20260910-dameli-donors.csv', errors, file_num=999,
        param_counter={})
    expected = json.loads((RAW/'20260910-dameli-donors-audit.json').read_text())
    assert not errors.getvalue()
    # the audited donor spellings are the Original; the display form is their house
    # transcription (w → v, š → ś), nothing else is re-interpreted
    assert [r.old_form for r in rows] == [r['Form'] for r in expected]
    assert [r.form for r in rows] == [
        r['Form'].replace('w', 'v').replace('š', 'ś') for r in expected]

def test_compiled_heads_when_requested():
    if not os.environ.get('DAMELI_COMPILED_CHECK'):
        pytest.skip('Run with DAMELI_COMPILED_CHECK=1 against a fresh compiled build')
    aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
    forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open())}
    for r in json.loads((RAW/'20260910-dameli-donors-audit.json').read_text()):
        f=forms[aliases[r['ID']]]
        assert f['Language_ID']==r['Language_ID'] and f['Form']==r['Form']
        assert f['Status'] != 'unlinked'
        assert f['Native']==r['Native']
        assert set(r['Tags'].split()) <= set(f['Tags'].split())
