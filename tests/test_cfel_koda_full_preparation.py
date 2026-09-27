"""Full-source staged recovery; these tests do not install or build a database."""
import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest
from segments import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/forms/raw_data'


def prepare(package):
    path = RAW / package / 'prepare_full.py'
    spec = importlib.util.spec_from_file_location(package + '_full', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


@pytest.mark.parametrize('package,count,profile', [
    ('cfel_koda_2022', 3230, 'cfel-koda-print'),
    ('cfel_koda_api_2026', 3223, 'cfel-koda-api'),
])
def test_complete_source_accounting_and_literal_profile(package, count, profile):
    rows, audit = prepare(package)
    assert len(rows) == count and len(audit) == 2450
    assert rows == list(csv.reader((RAW / package / 'full-proposal.csv').open()))
    keys = {r[10] for r in rows}
    assert len(keys) == count
    assert keys == {key for unit in audit for key in unit['emitted_keys']}
    tokenizer = Tokenizer(str(ROOT / 'conversion' / (profile + '.txt')))
    for row in rows:
        assert len(row) == 15 and row[0] == 'Koda' and row[2] and row[4]
        assert row[2] == unicodedata.normalize('NFC', row[2])
        assert not row[11] or row[11] in keys
        assert not any(row[i] for i in [1, 5, 8, 9, 12, 13])
        converted = tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' ')
        assert converted and '�' not in converted, (row[10], row[2], converted)
        if row[2] == row[4]:
            assert unicodedata.normalize('NFC', converted) == ' '.join(row[2].split())
    pilot = '20260925-cfel-koda-print-pilot' if package == 'cfel_koda_2022' else '20260925-cfel-koda-adornments'
    prior = list(csv.reader((RAW / package / (pilot + '.csv')).open()))
    assert {r[10] for r in prior} <= keys
    assert rows == list(csv.reader((ROOT / f'data/other/forms/{pilot}.csv').open()))


def test_independent_audit_hashes_and_scoped_parser():
    import hashlib
    import io
    import make_cldf
    report = json.loads((RAW / 'cfel_koda_2022/independent-full-audit-20260926-pass1.json').read_text())
    assert len(report['sample']) == 20 and all(x['status'] == 'pass' for x in report['sample'])
    for path, digest in report['sha256'].items():
        local = RAW / Path(path).parent.name / Path(path).name
        assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
    for stem, count in [('20260925-cfel-koda-print-pilot', 3230), ('20260925-cfel-koda-adornments', 3223)]:
        path = ROOT / f'data/other/forms/{stem}.csv'
        raw = {r[10]: r for r in csv.reader(path.open())}
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(path), errors, name=stem)
        assert len(parsed) == stats['converted'] == count and not errors.getvalue()
        assert {r.entry_key for r in parsed} == set(raw)
        assert all(r.old_form == raw[r.entry_key][2] and r.native == raw[r.entry_key][4] for r in parsed)


def test_native_only_variants_and_print_publisher_separation():
    print_rows, _ = prepare('cfel_koda_2022')
    api_rows, api_audit = prepare('cfel_koda_api_2026')
    assert sum(bool(x['emitted_keys']) for x in api_audit) == 2448
    assert all(r[7].startswith('cfel2026koda[') and ';' not in r[7] for r in api_rows)
    by_key = {r[10]: r for r in print_rows}
    taste = by_key['cfelkoda2022:p145:entry:2']
    assert taste[2] == 'sad̪' and taste[4] == 'স্বাদ'
    assert by_key['cfelkoda2022:p145:entry:2:variant:2'][2:6] == ['স্বরম', 'Taste', 'স্বরম', '']
    assert by_key['cfelkoda2022:p145:entry:2:variant:3'][2] == 'সিবিল'
    assert by_key['cfelkoda2022:p253:entry:5'][2] == 'manɖi t̪ihi'
    assert by_key['cfelkoda2022:p12:entry:3'][2] == 'lutur na'
    assert by_key['cfelkoda2022:p12:entry:3:variant:2'][2] == 'makuri'
    assert [by_key[f'cfelkoda2022:p153:entry:2:expansion:{i}'][2] for i in (1, 2, 3)] == ['bʰaʤi alu', 'bʰaʤi manɖi', 'bʰaʤi haku']


def test_component_transcription_never_impersonates_full_head():
    rows, audit = prepare('cfel_koda_2022')
    by_key = {r[10]: r for r in rows}
    for key in ['p13:entry:6', 'p61:entry:5', 'p69:entry:8', 'p176:entry:5', 'p130:entry:5']:
        row = by_key['cfelkoda2022:' + key]
        assert row[2] == row[4] and not row[5]
        assert 'Source transcription scope:' in row[6]
        assert 'uncertain' in row[14].split()
    for key in ['p312:entry:2', 'p303:entry:4', 'p306:entry:2']:
        row = by_key['cfelkoda2022:' + key]
        assert row[2] == row[4] and not row[5]
    assert all(x['semantic_review']['decision'] for x in audit)


def test_all_visual_reviews_and_source_script_repairs_are_applied():
    package = RAW / 'cfel_koda_2022'
    ipa = json.loads((package / 'native-ipa-visual-review.json').read_text())
    assert len(ipa) == 2126
    rows, audit = prepare('cfel_koda_2022')
    by_native = {(x['source_unit']['native_xps_page'], x['source_unit']['native_item']): x for x in audit}
    for position, expected in [((287, 2), 'হুডিংবা'), ((335, 4), 'হুডিং হেরেলহন'), ((218, 1), 'সাঞ্‌জু')]:
        assert by_native[position]['readings'][0]['row'][4] == expected
    assert by_native[(14, 1)]['readings'][0]['row'][2] == 'ʈæp'
    assert all('‘Cardinal' not in r[6] for r in rows)
