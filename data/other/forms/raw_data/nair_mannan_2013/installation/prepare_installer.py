"""Produce a standalone four-target installer from the tested transaction template.

This preparation writes only this package; no source/registry assets are installed.
The generated installer has no runtime dependency on the template package.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
TEMPLATE = DATA / 'data/other/forms/raw_data/nair_malaveda_2013/installation/install_source.py'
assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == 'af3eaa24d0499b754860af8090e580bdd95831e77f46ee0f980b9d5e392bcca5'
assert not (HERE / 'root-acceptance.json').exists()
text = TEMPLATE.read_text().replace('Malaveda', 'Mannan').replace('malaveda', 'mannan')
text = text.replace('six-target', 'four-target').replace('223', '154').replace('240', '140')
old = "    'languages': 'cldf/languages.csv', 'dialects': 'cldf/dialects.csv', 'bib': 'cldf/sources.bib'}"
assert old in text
text = text.replace(old, "    'bib': 'cldf/sources.bib'}")
start, end = text.index('def validate_assets('), text.index('\ndef shared_membership():')
text = text[:start] + '''def validate_assets(approval, planned):
    rows = list(csv.reader(io.StringIO(planned['csv'].decode())))
    assert len(rows) == approval['rows'] == 154
    assert all(len(r) == 15 and r[0] in {'Mannan', 'Malayalam'} and r[2] and r[10] for r in rows)
    assert sum(r[0] == 'Mannan' for r in rows) == 123
    assert sum(r[0] == 'Malayalam' for r in rows) == 31
    keys = {r[10] for r in rows}
    assert len(keys) == 154
    for row in rows:
        assert not any(row[i] for i in (1, 4, 5, 8, 11, 12, 13))
        assert unicodedata.is_normalized('NFC', row[2])
        assert all(unicodedata.category(c) != 'Co' and c != '\\ufffd' for c in row[2])
        assert row[7].startswith('nair2013mannan[p. ')
    audit = list(map(json.loads, planned['audit'].decode().splitlines()))
    assert len(audit) == len({a['audit_key'] for a in audit}) == approval['audit_units'] == 140
    audit_keys = [k for a in audit for k in a['entry_keys']]
    assert len(audit_keys) == len(set(audit_keys)) == 154 and set(audit_keys) == keys
    assert all(a['accepted_for_installation'] is True and 'source_record' in a for a in audit)
    assert sum(len(a['source_record']['witness_dispositions']) for a in audit) == 208
    assert sum(w['disposition'] == 'audit_only' for a in audit for w in a['source_record']['witness_dispositions']) == 54
    settings = yaml.safe_load(planned['yaml'])
    assert settings['file'] == STEM + '.csv'
    assert settings['defaults']['transcription'] == {'convert': False, 'preserve_hyphens': True}
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['forms']['split_alternates'] is False
    assert settings['defaults']['gloss']['grammar_tags'] is False
    assert settings['defaults']['importer']['commands'] == [['data/other/forms/raw_data/nair_mannan_2013/installation/install_source.py', '--install']]
    refs = parse_string((PACKAGE / 'installation/staged-references.bib').read_text(), 'bibtex').entries
    assert set(settings['sources']) == set(refs) == {'nair2013mannan', 'joseph1980devikolam'}
    for key, source in settings['sources'].items():
        assert source['reference'] == {'editor': 'OpenAI Codex', 'ocr': False,
            'etymology_provenance': 'source' if key == 'nair2013mannan' else 'none'}
    meta = SourceMeta([PACKAGE / 'installation/20260927-nair-mannan.yaml'])
    assert set(meta.sources) == set(refs)
    installed_bib = parse_string(planned['bib'].decode(), 'bibtex').entries
    assert all(installed_bib[k] == v for k, v in refs.items())

''' + text[end:]
start, end = text.index('def shared_membership('), text.index('\ndef plan(')
text = text[:start] + '''def shared_membership():
    expected = parse_string((PACKAGE / 'installation/staged-references.bib').read_text(), 'bibtex').entries
    actual = parse_string((DATA / TARGETS['bib']).read_text(), 'bibtex').entries
    assert all(k in actual and actual[k] == v for k, v in expected.items()), 'shared bibliography contribution drift'

''' + text[end:]
text = text.replace("'targets': 6", "'targets': 4")
old_hash = "def file_hash(path):\n    return digest(path.read_bytes()) if path.exists() else None"
assert old_hash in text
text = text.replace(old_hash, '''def file_hash(path):
    if not path.exists():
        return None
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()''')
destination = HERE / 'install_source.py'
destination.write_text(text)
print('Standalone four-target Mannan installer prepared; no canonical writes')
