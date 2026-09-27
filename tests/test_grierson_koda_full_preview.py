"""Whole-source Koda proposal checks; no canonical installation or DB build."""
import csv
import importlib.util
import io
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_koda_birbhum_1906'
spec = importlib.util.spec_from_file_location('koda_full_preview', PACKAGE / 'prepare_full_preview.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def staged():
    with (PACKAGE / 'full-preview.csv').open() as source:
        return list(csv.reader(source))


def test_complete_accounting_and_identity():
    rows, audit = module.build()
    assert rows == staged()
    assert audit == [json.loads(line) for line in (PACKAGE / 'full-preview-audit.jsonl').read_text().splitlines()]
    assert len(rows) == 654 and len(audit) == 735
    assert Counter(r['section'] for r in audit) == {
        'grammar': 41, 'birbhum': 333, 'bankura': 93, 'prose': 27, 'table': 241,
    }
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    legacy = json.loads((PACKAGE / 'whole-source-assembly-plan-20260926.json').read_text())['legacy_key_mapping']
    assert set(legacy.values()) <= keys
    assert Counter(r[14].split()[0] for r in rows) == {
        module.LECTS['Birbhum']: 256, module.LECTS['Bankura']: 96, module.LECTS['Dhangar']: 302,
    }
    for unit in audit:
        assert set(unit['entry_keys'] + unit['reuse_entry_keys']) <= keys
    assert len([a for a in audit if a['status'] == 'physical_continuation']) == 5
    assert {a['source_reading']['prompt_number'] for a in audit if a['section'] == 'table'} == set(range(1, 242))


def test_abbreviations_uncertainty_and_source_qualification():
    rows, audit = module.build()
    by_key = {r[10]: r for r in rows}
    assert {'copula', '1sg', 'pret'} <= set(by_key['grierson1906koda:grammar:p110:copula02'][14].split())
    assert {'pron', '3sg'} <= set(by_key['grierson1906koda:grammar:p110:pronoun01'][14].split())
    for unit in audit:
        raw = unit['source_reading']
        emitted = [by_key[k] for k in unit['entry_keys'] + unit['reuse_entry_keys']]
        if raw.get('uncertainty'):
            assert emitted and all('uncertain' in r[14].split() for r in emitted)
        if unit['lect'] == 'Bankura':
            assert all('editorially restored' in r[6] for r in emitted)
        if unit['section'] == 'table' and raw['prompt_number'] in {103, 104}:
            assert len(emitted) == 1
            assert emitted[0][2] == unicodedata.normalize('NFC', raw.get('form_candidate', raw.get('form')))
            assert 'No missing phrase text supplied' in emitted[0][6]
    assert all(not r[2].endswith('-') for r in rows)
    assert all(not r[5] and not any(r[11:14]) for r in rows)


def test_preservation_profile_and_scoped_parser():
    import make_cldf
    rows = staged()
    tokenizer = Tokenizer(str(PACKAGE / 'full-preservation-profile.txt'))
    for row in rows:
        displayed = unicodedata.normalize('NFC', tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' '))
        assert displayed == row[2].replace('w', 'v').replace('ṅ', 'ŋ')
    key = 'grierson-koda-birbhum-1906'
    previous = make_cldf.convertors.get(key)
    try:
        make_cldf.convertors[key] = tokenizer
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(PACKAGE / 'full-preview.csv'), errors, name='20260925-grierson-koda-birbhum')
        assert not errors.getvalue()
        assert len(parsed) == stats['converted'] == 654
        assert {r.entry_key: r.old_form for r in parsed} == {r[10]: r[2] for r in rows}
        assert all(not r.ipa for r in parsed)
    finally:
        if previous is None:
            make_cldf.convertors.pop(key, None)
        else:
            make_cldf.convertors[key] = previous


def test_source_metadata_repairs_and_no_lexical_changes():
    import tags
    rows = staged()
    with (PACKAGE / 'full-preview-pass1.csv').open() as old:
        previous = list(csv.reader(old))
    assert len(previous) == len(rows)
    for before, after in zip(previous, rows):
        assert all(before[i] == after[i] for i in range(15) if i not in (6, 14))
        assert set(after[14].split()[1:]) <= tags.GRAMMATICAL_TAGS | tags.GENDER_TAGS
    by_key = {r[10]: r for r in rows}
    _, audit = module.build()
    for unit in audit:
        raw = unit['source_reading']
        emitted = [by_key[k] for k in unit['entry_keys'] + unit['reuse_entry_keys']]
        if unit['section'] == 'grammar' and unit['source_cell_key'].endswith(('number06', 'number07', 'number08', 'number09', 'number10')):
            assert emitted and all({'num', 'loanword'} <= set(r[14].split()) for r in emitted)
            assert all('Grierson explicitly classifies' in r[6] for r in emitted)
        if unit['section'] == 'bankura' and '(sic)' in raw.get('note', ''):
            assert emitted and all('(sic)' in r[6] for r in emitted)
        if unit['section'] == 'table':
            number = raw['prompt_number']
            if number in (133, 134, 136, 137):
                degree = 'comparative' if number in (133, 136) else 'superlative'
                assert all('degree' in r[14].split() and degree in r[6] for r in emitted)
            required = {162: {'1sg', 'pret'}, 180: {'2sg', 'pres'}, 183: {'2pl', 'pres'}, 192: {'1sg', 'pret', 'progressive'}, 215: {'2pl', 'pret'}}.get(number)
            if required:
                assert emitted and all(required <= set(r[14].split()) for r in emitted)
            if number in (215, 216, 217, 218, 219):
                assert all('multiword-expression' not in r[14].split() for r in emitted)
