import csv
import importlib.util
import io
import json
import unicodedata as ud
from pathlib import Path

from make_cldf import parse_file
from segments.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/forms/raw_data/peterson_turi_2024'
STEM = '20260921-peterson-turi'
SPEC = importlib.util.spec_from_file_location('peterson_turi', RAW / 'import_source.py')
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def rows():
    return list(csv.reader((ROOT / 'data/other/forms' / f'{STEM}.csv').open()))


def records():
    return json.loads((RAW / 'records.json').read_text())


def convert(text):
    tokenizer = Tokenizer(str(ROOT / 'conversion/peterson-turi.txt'))
    return ud.normalize('NFC', tokenizer(text, column='IPA').replace(' ', '').replace('#', ' '))


def test_complete_source_accounting_and_reproduction():
    actual = rows()
    rebuilt, audit = IMPORTER.build(records())
    assert actual == rebuilt
    assert audit == json.loads((RAW / f'{STEM}-audit.json').read_text())
    assert len(records()) == len(audit) == 275
    assert sum(a['status'] == 'unelicited' for a in audit) == 51
    assert len(actual) == len({r[10] for r in actual}) == 246
    assert sum('IA' in a['raw_form'] for a in audit) == 106
    assert all(len(r) == 15 and r[2] and r[3] for r in actual)
    assert all(ud.is_normalized('NFC', f) and '�' not in f for r in actual for f in r)
    assert all(not r[1] and not r[8] and not r[11] and not r[12] and not r[13] for r in actual)


def test_printed_typos_subscripts_and_page_identity():
    indexed = {r['entry_key']: r for r in records()}
    assert indexed['peterson2024turi:p297:item234']['gloss'] == 'why'
    assert indexed['peterson2024turi:p296:item234']['gloss'] == 'they (f.)'
    assert indexed['peterson2024turi:p292:item52.1']['gloss'] == 'flour'
    assert indexed['peterson2024turi:p291:item17']['printed_item'] == '17,'
    actual = {r[10]: r for r in rows()}
    assert actual['peterson2024turi:p294:item138:form1'][3] == 'plain (2)'
    assert actual['peterson2024turi:p294:item137:form1'][3] == 'plain (1)'
    assert actual['peterson2024turi:p294:item107:form1'][3] == 'jar (2)'


def test_diacritics_remain_in_their_source_rows():
    indexed = {r['entry_key']: r for r in records()}
    for item, form in [('40', 'pʰuhuɽi'), ('41', 'sɔ̃dɔrɔ'), ('57', 'sibil'), ('58', 'sɔ̃ʊ̃̃')]:
        assert indexed[f'peterson2024turi:p292:item{item}']['raw_form'] == form
    assert indexed['peterson2024turi:p293:item83']['raw_form'] == 'tidʒu'
    assert indexed['peterson2024turi:p293:item84']['raw_form'] == 'rɔ̃'


def test_alternatives_annotation_scope_and_no_inferred_borrowing_edges():
    actual = {r[10]: r for r in rows()}
    assert [actual[f'peterson2024turi:p294:item117:form{i}'][2] for i in range(1, 4)] == ['ʈɑkɑ', 'pɑisɑ', 'kɛtʃɑ']
    assert actual['peterson2024turi:p297:item247:form1'][2] == 'ɖihi'
    assert 'IA, Magadhan' in actual['peterson2024turi:p297:item247:form1'][9]
    assert 'likely IA' in actual['peterson2024turi:p296:item206:form1'][9]
    assert 'uncertain' in actual['peterson2024turi:p296:item206:form1'][14]
    assert 'pl' in actual['peterson2024turi:p296:item232:form1'][14].split()
    assert all('IA' not in r[2] and '(' not in r[2] for r in actual.values())


def test_house_profile_complete_coverage_and_difficult_sequences():
    assert convert('ʈʰuɽʱi') == 'ṭʰuṛʰi'
    assert convert('mɑjɑŋ') == 'mayaŋ'
    assert convert('dʒɔhɑ') == 'jɔha'
    assert convert('tʃiʈʈʰi') == 'ciṭṭʰi'
    assert convert('gɑr̥ɑ') == 'gar̥a'
    assert convert('sɔ̃ʊ̃̃') == 'sɔ̃ũ'
    assert convert('ʈʰə') == 'ṭʰə'
    for row in rows():
        assert '�' not in convert(row[2])
        assert convert(ud.normalize('NFC', row[2])) == convert(ud.normalize('NFD', row[2]))


def test_registered_dialect_and_parse_file_layers():
    dialect = next(r for r in csv.DictReader((ROOT / 'cldf/dialects.csv').open())
                   if r['Tag'] == IMPORTER.DIALECT)
    assert dialect['Language_ID'] == 'Turi'
    assert not dialect['Latitude'] and not dialect['Longitude']
    error = io.StringIO()
    parsed, stats = parse_file(str(ROOT / 'data/other/forms' / f'{STEM}.csv'), errors=error)
    assert not error.getvalue()
    assert stats == {'converted': 246, 'for_conversion': 246}
    assert len(parsed) == 246
    original = {r[10]: r for r in rows()}
    for row in parsed:
        raw = original[row.entry_key]
        assert row.old_form == raw[2] and row.ipa == raw[5]
        assert row.form == convert(raw[2])


def test_compiled_source_survival_and_no_edges():
    # Mandatory full-build gate, deliberately fails against the stale pre-ingest CLDF.
    expected = {r[10] for r in rows()}
    source_keys = [r for r in csv.DictReader(
        (ROOT / 'cldf/form-source-keys.csv').open())
        if r['Source_Key'] in expected]
    legacy = {r['Legacy_ID'] for r in source_keys}
    aliases = {r['Legacy_ID']: r['Form_ID'] for r in csv.DictReader(
        (ROOT / 'cldf/form-id-aliases.csv').open()) if r['Legacy_ID'] in legacy}
    keys = {r['Source_Key']: aliases.get(r['Legacy_ID'], r['Legacy_ID']) for r in source_keys}
    assert set(keys) == expected
    compiled = {r['ID']: r for r in csv.DictReader((ROOT / 'cldf/forms.csv').open())
                if 'peterson2024turi' in r['Source']}
    assert set(keys.values()) == set(compiled)
    assert all(r['Status'] == 'unlinked' and r['Language_ID'] == 'Turi' for r in compiled.values())
    for row in rows():
        built = compiled[keys[row[10]]]
        assert built['Form'] == convert(row[2])
        assert built['Original'] == row[2] and built['Phonemic'] == row[5]
        assert row[7] in built['Source']
    assert not any(r['Child_ID'] in compiled for r in csv.DictReader((ROOT / 'cldf/edges.csv').open()))
    assert any(r['ID'] == 'peterson2024turi' for r in csv.DictReader((ROOT / 'cldf/references.csv').open()))
