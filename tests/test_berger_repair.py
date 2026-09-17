"""Source-backed regression checks for the September Berger repair."""
import csv
import gzip
import json
from pathlib import Path

from segments import Tokenizer

from test_berger_cleanup import cleanup as b

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'data/other/forms/raw_data/berger_2026'


def installed():
    return [r for path in (b.AUTO_OUTPUT, b.GOLD_OUTPUT) for r in csv.reader(path.open())]


def test_shifted_scan_keeps_headword_and_indented_definition_in_same_column():
    page = next(p for p in b.load_pages(b.CACHE_DIR) if p['pdf_page'] == 32)
    head = next(l for l in page['lines'] if l['text'].startswith('bot -'))
    definition = next(l for l in page['lines'] if l['text'].startswith('bes. das Buddha'))
    assert b.physical_column(head['left'], page['width']) != b.physical_column(definition['left'], page['width'])
    centers = b.repair.column_starts(page)
    assert b.repair.column(head['left'], centers) == b.repair.column(definition['left'], centers) == 2
    units = b.reconstruct_units([page], corrected=True)
    article = next(u for u in units if u.text.startswith('bot -'))
    assert 'bes. das Buddha-Relief' in article.text
    assert not any(u.text.startswith('bes.') for u in units)


def test_surviving_keys_are_anchored_to_lines_not_recomputed_ordinals():
    pages = [p for p in b.load_pages(b.CACHE_DIR) if p['pdf_page'] in (32, 52, 79)]
    old = {(u.pdf_page, u.left, u.top): u.stable_key for u in b.reconstruct_units(pages)}
    current = b.reconstruct_units(pages, corrected=True)
    assert all(u.stable_key == old[(u.pdf_page, u.left, u.top)] for u in current
               if (u.pdf_page, u.left, u.top) in old)


def test_hanging_dazu_head_stays_with_its_definition():
    page = next(p for p in b.load_pages(b.CACHE_DIR) if p['pdf_page'] == 143)
    units = b.reconstruct_units([page], corrected=True)
    article = next(u for u in units if u.stable_key == 'berger:p278:c2:e007')
    assert 'geboren werden' in article.text
    assert 'Plattform' not in article.text
    assert 'ys. `man' not in article.text


def test_reviewed_class_plural_dialect_and_headword_scope():
    by = {r[10]: r for r in installed()}
    assert by['berger-entry-9341'][2:4] == ['trin', 'favour']
    assert {'noun', 'Burushaski-class-Y'} <= set(by['berger-entry-9341'][14].split())
    assert 'verb' not in by['berger-entry-9341'][14].split()
    for key in ['berger-entry-69', 'berger:p009:c2:e002', 'berger-entry-182']:
        assert 'pl' not in by[key][14].split()
        assert any(r[11] == key and 'pl' in r[14].split() for r in by.values())
    amin = [r for r in by.values() if r[10] == 'berger-entry-182' or r[11] == 'berger-entry-182']
    assert {'Burushaski-class-HM', 'Burushaski-class-HF', 'Burushaski-class-X', 'Burushaski-class-Y'} <= {
        t for r in amin for t in r[14].split()}
    assert by['berger-entry-5560'][2:4] == ['luṭhúri gaíṅ', 'a variety of grape']
    assert by['berger:p164:c1:e011'][2] == 'ġaáṣ'
    assert 'dialect:Bur:Berger-NG:Nager' in by['berger-entry-3734'][14].split()
    assert '-iṅ -ičaṅ ng.' in by['berger-entry-3734'][6]
    assert by['berger:p329:c1:e009'][2] == '-phíliṣ'
    assert by['berger-entry-1958'][2] == 'dáal :t-'
    assert 'abolish a tax' in by['berger-entry-1958'][3]
    assert 'berger:p097:c1:e014' not in by
    assert '²khaṣ' in by['berger-entry-5076'][6]
    assert by['berger:gold:cdial11406:balanees-man'][3]


def test_catalog_cannot_attach_current_wrong_articles_to_old_meanings():
    rows = {r[10]: r for r in installed()}
    catalog = list(csv.DictReader((ROOT / 'data/burushaski_cognates.csv').open()))
    keys = {key for c in catalog for key in c['Evidence_Keys'].split('|')}
    protection = json.loads((PACKAGE / 'catalog-protection.json').read_text())
    assert all(k + ':legacy-graph' in keys and k not in keys for k in protection['keys'])
    assert all(key in rows for key in keys if key.startswith(('berger-', 'berger:')))
    for key in ('berger-entry-1712', 'berger-entry-2920', 'berger-entry-7051', 'berger-entry-7630'):
        assert key in rows and key not in keys
        assert rows[key][2] != rows[key + ':legacy-graph'][2]
    # Catalog graph construction resolves the protected keys, not just their presence.
    from burushaski_cognates import apply_catalog
    subset = [c for c in catalog if c['Set_ID'] in protection['set_ids']]
    needed = {k for c in subset for k in c['Evidence_Keys'].split('|')}
    forms = [[k, rows[k][0] if k in rows else 'Bur', rows[k][2] if k in rows else 'fixture',
              rows[k][3] if k in rows else '', '', '', '', '', '', '', 'berger', '', '', 'local', '', '', '']
             for k in needed]
    apply_catalog(forms, {k: k for k in needed}, subset)
    assert all(r[11].startswith('pbsk-') for r in forms)


def test_all_rows_have_audits_resolved_endpoints_registered_tags_and_profiles():
    rows = installed()
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    assert {len(r) for r in rows} == {15}
    assert {r[0] for r in rows} == {'Bur'}
    assert all(target in keys for r in rows for col in (11, 12, 13) for target in r[col].split('|') if target)
    with gzip.open(PACKAGE / 'audit.jsonl.gz', 'rt') as stream:
        audit = list(map(json.loads, stream))
    assert {k for a in audit for k in a['Emitted_Keys'].split('|') if k} == keys
    profile = Tokenizer(str(ROOT / 'conversion/berger.txt'))
    assert not [(r[10], r[2]) for r in rows if '�' in profile(r[2], column='IPA')]
    registry = {r['Tag'] for r in csv.DictReader((ROOT / 'cldf/dialects.csv').open())}
    assert {t for r in rows for t in r[14].split() if t.startswith('dialect:')} <= registry
    from tags import GRAMMATICAL_TAGS
    assert {t for r in rows for t in r[14].split() if t.startswith('Burushaski-class-')} <= GRAMMATICAL_TAGS
    parents = {r[10]: r[11] for r in rows if r[11]}
    for key in parents:
        seen = set()
        while key in parents:
            assert key not in seen, f'variant cycle at {key}'
            seen.add(key)
            key = parents[key]


def test_reproduction_inputs_are_pinned():
    b.repair.verify_inputs()
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert manifest['pdf_sha256'] == b.PDF_SHA256
    assert len(b.load_pages(b.CACHE_DIR)) == 241


def test_current_source_scope_counts_and_fresh_review_corrections():
    with gzip.open(PACKAGE / 'audit.jsonl.gz', 'rt') as stream:
        audit = list(map(json.loads, stream))
    summary = json.loads((PACKAGE / 'summary.json').read_text())
    assert len(audit) == summary['audit_rows']
    assert len(installed()) == summary['auto_rows'] + summary['gold_rows']
    assert {int(a['Printed_Page']) for a in audit if a['Printed_Page']} == set(range(9, 487))
    assert 51 not in {a['PDF_Page'] for a in audit}
    by_key = {r[10]: r for r in installed()}
    sample = json.loads((PACKAGE / 'fresh-sample-review.json').read_text())
    assert len(sample) == 20
    assert sum(r['material_error_before'] for r in sample) == 17
    for r in sample:
        if 'form' in r['reviewed_fix']:
            assert by_key[r['installed_key']][2] == r['reviewed_fix']['form']


def test_gold_cannot_copy_another_articles_grammar_or_assert_tentative_links():
    by = {r[10]: r for r in installed()}
    assert by['berger-entry-329'][3].startswith('needed, necessary')
    assert 'Stimme' not in by['berger-entry-329'][6]
    assert not by['berger-entry-329'][1] and not by['berger-entry-328'][1]
    assert by['berger-entry-386'][2] == 'baát'
    assert by['berger-entry-386'][3].startswith('porridge')
    assert by['berger-entry-386'][1] == '9331'
    assert 'Burushaski-class-Y' in by['berger-entry-334-dialect-1'][14]
    assert 'Berger-NG:Nager' in by['berger-entry-450-dialect-1'][14]
    assert not by['berger-entry-27'][1]
    assert 'T 11433' in by['berger-entry-27'][6]
    assert not b.direct_turner_ids('(T 1197 oder 1221)', {'1197', '1221'})
    assert by['berger:p024:c1:e009:source-article'][2:4] == ['awáaz', 'voice']


def test_every_retired_source_key_has_a_reviewable_redirect():
    with gzip.open(PACKAGE / 'installed-before.csv.gz', 'rt') as stream:
        before = {r[10] for r in csv.reader(stream)}
    now = {r[10] for r in installed()}
    aliases = list(csv.DictReader((ROOT / 'data/other/form_aliases/20260914-berger.csv').open()))
    assert {r['Retired_Source_Key'] for r in aliases} == before - now
    assert all(r['Target_Source_Key'] in now and r['Reason'] for r in aliases)
    assert len({r['Retired_Source_Key'] for r in aliases}) == len(aliases)


def test_source_conversion_preserves_every_key_and_grammatical_label(monkeypatch):
    import io
    import make_cldf
    monkeypatch.chdir(ROOT)
    for path in (b.AUTO_OUTPUT, b.GOLD_OUTPUT):
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(path), errors, name='berger')
        original = {r[10]: r for r in csv.reader(path.open())}
        assert not errors.getvalue()
        assert {r.entry_key for r in parsed} == original.keys()
        assert stats['converted'] == stats['for_conversion'] == len(parsed)
        assert all(r.old_form == original[r.entry_key][2] for r in parsed)
        assert all(set(original[r.entry_key][14].split()) <= set(r.tags.split()) for r in parsed)
        assert all(r.notes == original[r.entry_key][6] for r in parsed)


def test_durable_ids_and_aliases_against_current_berger_registry():
    from assign_form_ids import assign_ids
    from source_key_aliases import apply_source_key_aliases
    with (ROOT / 'data/form-identities.csv').open() as stream:
        registry = [r for r in csv.DictReader(stream)
                    if r['Source_Key'].startswith(('berger:', 'berger-'))]
    rows = installed()
    forms = [dict(ID=f'berger-check-{i}', Language_ID=r[0], Form=r[2], Original=r[2],
                  Gloss=r[3], Source=r[7]) for i, r in enumerate(rows)]
    keys = {r['ID']: source[10] for r, source in zip(forms, rows)}
    mapping, next_registry = assign_ids(forms, registry, keys)
    current = {r['Source_Key']: r['Form_ID'] for r in next_registry if r['Status'] == 'active'}
    previous = {r['Source_Key']: r['Form_ID'] for r in registry if r['Status'] == 'active'}
    assert all(current[k] == old_id for k, old_id in previous.items() if k in current)
    aliases = {}
    apply_source_key_aliases(aliases, registry, next_registry, set(mapping.values()),
                            paths=[ROOT / 'data/other/form_aliases/20260914-berger.csv'])
    retired = list(csv.DictReader((PACKAGE / 'aliases.csv').open()))
    for r in retired:
        if r['Retired_Source_Key'] in previous:
            assert aliases[previous[r['Retired_Source_Key']]] == current[r['Target_Source_Key']]
