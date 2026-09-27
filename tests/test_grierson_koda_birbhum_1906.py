"""Installed whole-source LSI IV Koda checks; no DB build."""
import csv
import importlib.util
import io
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_koda_birbhum_1906'
CSV = DATA / 'data/other/forms/20260925-grierson-koda-birbhum.csv'
spec = importlib.util.spec_from_file_location('koda_full_installed', PACKAGE / 'import_source_full.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    with CSV.open() as handle:
        return list(csv.reader(handle))


def test_installed_reproduces_full_audit_and_preserves_legacy():
    rows, audit = source.build()
    assert rows == installed() and len(rows) == 654 and len(audit) == 735
    assert audit == [json.loads(x) for x in (PACKAGE / 'full-installed-audit.jsonl').read_text().splitlines()]
    assert CSV.read_bytes() == (PACKAGE / 'full-preview.csv').read_bytes()
    old = list(csv.reader((PACKAGE / 'legacy-pilot.csv').open()))
    assert len(old) == 5 and {r[10] for r in old} <= {r[10] for r in rows}
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert manifest['installed'] == 654 and manifest['physical_units'] == 735
    assert 'deferred' in manifest['status']
    keyed = {r[10]: r for r in rows}
    prefix = 'grierson1906lsi4:koda_birbhum:'
    assert keyed[prefix + 'p109:ex01'][2:4] == ['hā̂ṛā̂', 'man']
    assert keyed[prefix + 'p109:ex07'][2:4] == ['lēl', 'see']
    assert keyed[prefix + 'p110:ex06'][2:4] == ['ãṭ', 'eight']


def test_all_lects_and_references_are_registered():
    with (DATA / 'cldf/dialects.csv').open() as handle:
        registry = {r['Tag']: r for r in csv.DictReader(handle)}
    rows = installed()
    seen = set()
    for row in rows:
        assert len(row) == 15 and row[0] == 'Koda'
        dialect = row[14].split()[0]
        seen.add(dialect)
        entry = registry[dialect]
        assert entry['Language_ID'] == 'Koda'
        assert not entry['Latitude'] and not entry['Longitude'] and not entry['Glottocode']
        assert all(c.startswith('grierson1906lsi4[p. ') and c.endswith(']') for c in row[7].split(';'))
        assert not row[5] and not any(row[11:14])
    assert len(seen) == 3
    bib = (DATA / 'cldf/sources.bib').read_text()
    assert bib.count('@book{grierson1906lsi4,') == 1
    assert '735 physical units yield 654 forms' in bib


def test_installed_profile_and_yaml_parse_all_originals():
    import make_cldf
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name='20260925-grierson-koda-birbhum')
    assert not errors.getvalue()
    assert len(parsed) == stats['converted'] == 654
    assert {r.entry_key: r.old_form for r in parsed} == {r[10]: r[2] for r in installed()}
    assert all(not r.ipa for r in parsed)
