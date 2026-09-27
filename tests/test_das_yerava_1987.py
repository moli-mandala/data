"""Acquisition-stage checks; these do not certify an installed ingestion."""
import csv
import hashlib
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/das_yerava_1987'


def rows():
    with (RAW / 'reviewed.tsv').open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def test_complete_two_page_table_and_source_positions():
    table = rows()
    assert len(table) == 42
    assert [(r['Page'], int(r['Item'])) for r in table] == (
        [('65', n) for n in range(1, 28)] + [('66', n) for n in range(1, 16)])
    assert sum(len(r[lect].split(', ')) for r in table
               for lect in ['Panjiri Yerava', 'Pani Yerava']) == 90
    assert all(r['English'] and r['Panjiri Yerava'] and r['Pani Yerava'] for r in table)


def test_embedded_ocr_evidence_is_pinned_and_not_silently_corrected():
    snapshot = json.loads((RAW / 'evidence/snapshot.json').read_text())
    assert snapshot['pdf_pages'] == 193
    assert 'Year of Publication = 1987' in snapshot['pdf_metadata']['Keywords']
    assert len(snapshot['pages']) == 13
    for page in snapshot['pages']:
        assert hashlib.sha256((RAW / 'evidence' / page['path']).read_bytes()).hexdigest() == page['sha256']
    ocr = (RAW / 'evidence/pdf-097.txt').read_text()
    assert 'Amma, Awa' in ocr and 'Ph-e' in ocr
    table = {r['English']: r for r in rows()}
    assert table['Mother']['Panjiri Yerava'] == 'Amma, Avva'
    assert table['House']['Pani Yerava'] == 'Pire'
    assert table['Firewood']['Pani Yerava'] == 'Kodu'


def test_unusual_printed_kinship_scope_and_distinct_prompts_survive():
    table = rows()
    assert table[2]['English'] == "Mother's Father; Father's Father; Mother's Mother"
    assert table[2]['Panjiri Yerava'] == 'Achcha'
    assert table[3]['English'] == "Father's Mother"
    assert table[26]['English'] == 'Arrack' and table[26]['Pani Yerava'] == 'Kallu'
    assert table[27]['English'] == 'Toddy' and table[27]['Panjiri Yerava'] == 'Kallu'


def importer():
    import importlib.util
    spec = importlib.util.spec_from_file_location('das_yerava', RAW / 'import_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_rows_audit_and_actual_pipeline_preserve_source_spelling():
    import io
    from collections import Counter
    from make_cldf import parse_file
    source = importer()
    expected, audit = source.build()
    path = source.ROOT / f'data/other/forms/{source.STEM}.csv'
    with path.open() as stream:
        assert list(csv.reader(stream)) == expected
    assert Counter(row[0] for row in expected) == {'Ravula': 48, 'Paniya': 42}
    assert len(audit) == 84
    errors = io.StringIO()
    parsed, stats = parse_file(str(path), errors=errors)
    assert not errors.getvalue()
    assert len(parsed) == 90 and stats == {'converted': 0, 'for_conversion': 0}
    original = {r[10]: r for r in expected}
    for row in parsed:
        assert row.form == row.old_form == original[row.entry_key][2]
        assert not row.ipa
    assert all(not r[i] for r in expected for i in [1, 4, 5, 8, 9, 11, 12, 13])
    assert all(len(r) == 15 and '�' not in ''.join(r) for r in expected)


def test_acceptance_hashes_and_registered_lects(tmp_path):
    source = importer()
    sample = source.generate(tmp_path)
    results = json.loads((RAW / 'acceptance-results.json').read_text())
    assert results['material_errors'] == 0 and results['sample'] == sample
    with (source.ROOT / 'cldf/dialects.csv').open() as stream:
        dialects = {r['ID']: r for r in csv.DictReader(stream)}
    for lect, (language, dialect) in source.LECTS.items():
        registered = dialects[dialect]
        assert registered['Language_ID'] == language
        assert registered['Source_Language_ID'] == lect
        assert not registered['Latitude'] and not registered['Longitude']
    from pybtex.database import parse_file
    import pybtex
    entry = parse_file(str(source.ROOT / 'cldf/sources.bib')).entries[source.SOURCE]
    assert entry == parse_file(str(RAW / 'source.bib')).entries[source.SOURCE]
    formatted = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
    assert '1987' in formatted and 'Kodagu' in formatted


def test_compiled_source_survival():
    source = importer()
    expected = {r[10] for r in source.build()[0]}
    with (source.ROOT / 'cldf/form-source-keys.csv').open() as stream:
        found = {r['Source_Key'] for r in csv.DictReader(stream) if r['Source_Key'] in expected}
    assert found == expected, 'Full-build gate: all 90 Das Yerava keys must survive compilation'
