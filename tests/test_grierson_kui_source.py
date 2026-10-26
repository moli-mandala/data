"""Installed complete Kui source-stage contracts; no database or full build."""
import collections
import csv
import hashlib
import io
import importlib.util
import json
import unicodedata as ud
from pathlib import Path
from urllib.parse import unquote

import make_cldf
import make_refs
import source_meta
import pybtex
import pytest
import yaml
from pybtex.database import parse_file
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'data/other/forms/raw_data/grierson_kui_1906'
INTEGRATION = PACKAGE / 'integration'
CANONICAL = ROOT / 'data/other/forms/20260927-grierson-kui.csv'
SOURCE = 'grierson1906kui'


def streamed_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exact_approved_assets():
    """Historical bytes remain authoritative until the exact addendum installs."""
    addendum = PACKAGE / 'source-printed-amd-addendum'
    receipt_path = addendum / 'correction-receipt.json'
    if not receipt_path.exists():
        return INTEGRATION / 'combined-candidate.csv', INTEGRATION / 'combined-audit.jsonl'
    spec = importlib.util.spec_from_file_location('kui_approved_spelling_addendum', addendum / 'apply_correction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    approval_path, approval = module.validate()
    receipt = json.loads(receipt_path.read_text())
    assert receipt['approval_sha256'] == streamed_hash(approval_path)
    assert set(receipt['output_sha256']) == {'csv', 'audit', 'importer'}
    assert receipt['output_sha256'] == {
        role: asset['output_sha256'] for role, asset in approval['targets'].items()
    }
    assert module.output_hashes(approval) == receipt['output_sha256']
    candidate = addendum / 'stage/corrected.csv'
    audit = addendum / 'stage/accepted-audit.jsonl'
    historical = (INTEGRATION / 'combined-candidate.csv').read_bytes().splitlines(keepends=True)
    corrected = candidate.read_bytes().splitlines(keepends=True)
    assert len(historical) == len(corrected) == 1832
    changes = [(old, new) for old, new in zip(historical, corrected) if old != new]
    assert len(changes) == 1
    old, new = changes[0]
    assert b'grierson1906kui:p566:outside-comparand:06' in old
    assert old.startswith(b'Kolami,,and,he,,,')
    assert new == old.replace(b'Kolami,,and,he,,,', b'Kolami,,amd,he,,,', 1)
    old_audit = (INTEGRATION / 'combined-audit.jsonl').read_bytes().splitlines(keepends=True)
    new_audit = audit.read_bytes().splitlines(keepends=True)
    assert len(old_audit) == len(new_audit) == 1859
    changes = [(json.loads(a), json.loads(b)) for a, b in zip(old_audit, new_audit) if a != b]
    assert len(changes) == 1
    before, after = changes[0]
    assert before['source_unit_key'] == 'grierson1906kui:p566:outside-comparand:06'
    assert after.pop('historical_source_record') == before['source_record']
    decision = after.pop('accepted_literal_addendum')
    assert (decision['old'], decision['accepted'], decision['stable_entry_key']) == (
        'and', 'amd', before['source_unit_key'])
    expected = json.loads(json.dumps(before))
    expected['source_record']['form'] = 'amd'
    assert after == expected
    return candidate, audit


def test_scoped_historical_w_policy(monkeypatch):
    import profile_policy
    profile = ROOT / 'conversion/grierson-kui-1906.txt'
    monkeypatch.setattr(profile_policy, 'profile_paths', lambda: [profile])
    assert profile_policy.audit({}) == {}
    for glyph in ('w', 'W'):
        assert ('grierson-kui-1906', glyph) in profile_policy.KEEP
        assert ('house', glyph) not in profile_policy.KEEP
        assert profile_policy.house_output(glyph, 'w', {'w': 'w'}, 'house') == 'v'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope='module')
def installed():
    rows = list(csv.reader(CANONICAL.open(newline='')))
    assert all(len(row) == 15 for row in rows)
    keyed = {row[10]: row for row in rows}
    assert len(rows) == len(keyed) == 1832
    return rows, keyed


def test_approved_assets_and_all_original_fields_survive_actual_parser(installed):
    rows, keyed = installed
    expected_csv, expected_audit = exact_approved_assets()
    assert CANONICAL.read_bytes() == expected_csv.read_bytes()
    assert (PACKAGE / 'installed-audit.jsonl').read_bytes() == expected_audit.read_bytes()
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CANONICAL), errors=errors)
    assert not errors.getvalue(), errors.getvalue()
    assert len(parsed) == stats['converted'] == 1832
    assert [r.entry_key for r in parsed] == [r[10] for r in rows]
    for r in parsed:
        raw = keyed[r.entry_key]
        assert (r.lang, r.param, r.cognateset, r.etymology) == tuple(raw[i] for i in (0, 1, 8, 9))
        assert (r.old_form, r.gloss, r.native, r.ipa, r.notes, r.source) == tuple(raw[i] for i in (2, 3, 4, 5, 6, 7))
        assert (r.variant_of_key, r.borrowed_from_key, r.derivation_parent_keys) == tuple(raw[i] for i in (11, 12, 13))
        assert r.tags == raw[14]
        expected = ud.normalize('NFC', raw[2]).lower().replace('ṅ', 'ŋ').strip('1234⁴5⁵67⁷,;.')
        assert ud.normalize('NFC', r.form) == expected, r.entry_key
        assert '\ufffd' not in r.form
        assert not r.native and not r.ipa
        assert set(make_refs.source_ids(raw[7])) == {SOURCE}


def test_complete_source_accounting_and_non_emitting_dispositions(installed):
    _, keyed = installed
    audit = jsonl(PACKAGE / 'installed-audit.jsonl')
    assert len(audit) == len({r['source_unit_key'] for r in audit}) == 1859
    assert collections.Counter(r['disposition'] for r in audit) == {
        'emitted': 1801, 'source-blank': 18, 'identity-hold': 15,
        'documentary': 8, 'reference-hold': 3,
        'reused-existing-occurrence': 12, 'joined-physical-fragment': 2,
    }
    assert sum(r['section'] == 'table' for r in audit) == 241
    assert sum(r['section'] == 'grammar' for r in audit) == 432
    for r in audit:
        if r['disposition'] in ('source-blank', 'identity-hold', 'documentary', 'reference-hold'):
            assert not r['entry_keys']
            assert r['source_unit_key'] not in keyed
        elif r['disposition'] != 'reused-existing-occurrence':
            assert set(r['entry_keys']) <= keyed.keys()
    # The ambiguous Kolami reference must not become three guessed lexical rows.
    for number in (12, 13, 14):
        key = f'{SOURCE}:p570:outside-comparand:{number:02}'
        record = next(r for r in audit if r['source_unit_key'] == key)
        assert record['disposition'] == 'reference-hold' and key not in keyed
    # Exact existing Naiki attestations retain their canonical identity and analysis.
    naiki = {r[10]: r for r in csv.reader((ROOT / 'data/other/forms/20260927-grierson-naiki.csv').open())}
    for r in audit:
        if r['disposition'] == 'reused-existing-occurrence':
            assert len(r['entry_keys']) == 1
            old = naiki[r['entry_keys'][0]]
            assert (old[0], old[2], old[3]) == ('Naiki', r['source_record']['form'], r['source_record']['gloss'])


def test_cross_references_alternatives_and_scoped_uncertainty(installed):
    _, keyed = installed
    audit = jsonl(PACKAGE / 'installed-audit.jsonl')
    table = {r['source_record']['prompt_id']: r for r in audit if r['section'] == 'table'}
    for number in (133, 136):
        assert table[number]['disposition'] == 'source-blank'
    for number in range(195, 201):
        present = table[number - 16]
        future = table[number]
        assert not future['physical_token']
        assert future['cross_reference_source_key'] == present['source_unit_key']
        assert [keyed[k][2] for k in future['entry_keys']] == [keyed[k][2] for k in present['entry_keys']]
        for k in future['entry_keys']:
            assert 'fut' in keyed[k][14].split() and 'pres' not in keyed[k][14].split()
            assert set(make_refs.source_ids(keyed[k][7])) == {SOURCE}
    scoped = [(652, 47, [False, True]), (652, 51, [False, True]),
              (660, 99, [True, False]), (672, 175, [False, False, True])]
    for page, number, flags in scoped:
        for i, uncertain in enumerate(flags, 1):
            row = keyed[f'{SOURCE}:p{page}:khondmals-table:prompt{number}:v{i}']
            assert ('uncertain' in row[14].split()) is uncertain
    for decision in json.loads((PACKAGE / 'root-grammar-alternative-emission-review-20260927.json').read_text())['decisions']:
        key = decision['source_unit_key']
        assert key not in keyed
        assert [keyed[f'{key}:v{i}'][2] for i in (1, 2)] == decision['forms']
    causal = keyed[f'{SOURCE}:p463:causal-example:2'][6]
    assert 'causal construction' in causal and 'causative' not in causal


def test_six_ordered_components_with_exact_endpoints(installed):
    rows, keyed = installed
    graph = json.loads((PACKAGE / 'root-component-relations-review-20260927.json').read_text())
    assert sum(len(r[13].split('|')) for r in rows if r[13]) == 6
    assert {r[10] for r in rows if r[13]} == {r['child_key'] for r in graph['accepted']}
    assert not any(r[11] or r[12] for r in rows)
    for decision in graph['accepted']:
        child = keyed[decision['child_key']]
        assert child[2] == decision['child_form']
        assert child[13].split('|') == decision['ordered_parent_keys']
        parents = [keyed[k] for k in child[13].split('|')]
        assert [p[2] for p in parents] == decision['parent_forms']
        assert all(p[0] == child[0] for p in parents)
        assert child[10] not in child[13].split('|')
    held = [r for r in json.loads((INTEGRATION / 'graph-accounting.json').read_text()) if not r['emitted_parent_keys']]
    assert len(held) == 5
    assert all(not keyed[r['child_key']][13] for r in held)


def test_all_physical_atoms_and_expressions_preserve_join_coverage(installed):
    _, keyed = installed
    atoms = jsonl(PACKAGE / 'specimens/atomic-emissions-reviewed.jsonl')
    physical = jsonl(PACKAGE / 'specimens/interlinear-reviewed.jsonl')
    assert len(atoms) == 888 and len(physical) == 890
    seen = []
    for r in atoms:
        row = keyed[r['emission_key']]
        assert (row[2], row[3]) == (r['emission_form'], r['emission_gloss'])
        seen.extend(r['physical_source_keys'])
        for tail in r['physical_source_keys'][1:]:
            assert tail not in keyed
    assert len(seen) == len(set(seen)) == 890
    assert set(seen) == {r['source_unit_key'] for r in physical}
    expressions = jsonl(PACKAGE / 'specimens/expression-emissions-reviewed.jsonl')
    assert len(expressions) == 77
    for r in expressions:
        row = keyed[r['source_unit_key']]
        assert (row[2], row[3]) == (r['emission_form'], r['emission_gloss'])
        assert set(r['atom_keys']) <= set(seen)
        assert set(r['tags']) <= set(row[14].split())


def test_registered_profile_references_dialects_and_settings(installed):
    rows, _ = installed
    languages = {r['ID'] for r in csv.DictReader((ROOT / 'cldf/languages.csv').open())}
    assert {r[0] for r in rows} <= languages
    assert (ROOT / 'conversion/grierson-kui-1906.txt').read_bytes() == (INTEGRATION / 'literal-preservation.txt').read_bytes()
    settings = yaml.safe_load(CANONICAL.with_suffix('.yaml').read_text())
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['transcription']['profile'] == 'grierson-kui-1906'
    assert settings['defaults']['forms']['split_alternates'] is False
    assert settings['sources'][SOURCE]['reference']['etymology_provenance'] == 'source'
    meta = source_meta.load()
    assert source_meta.importer_commands(meta, CANONICAL.stem) == [
        ['data/other/forms/raw_data/grierson_kui_1906/import_source.py', '--install']
    ]
    assert meta.flag(SOURCE, 'reference', 'etymology_provenance') == 'source'
    registry = {r['ID']: r for r in csv.DictReader((ROOT / 'cldf/dialects.csv').open())}
    expected = list(csv.DictReader((INTEGRATION / 'proposed-dialects.csv').open()))
    assert len(expected) == 4
    tags = {r['Tag'] for r in registry.values()}
    for r in expected:
        assert registry[r['ID']] == r
        assert not r['Latitude'] and not r['Longitude']
    for row in rows:
        for token in row[14].split():
            if token.startswith('dialect:'):
                assert token in tags
                assert unquote(token.split(':', 3)[1]) == row[0]
            else:
                assert token in GRAMMATICAL_TAGS | GENDER_TAGS
    bibliography = parse_file(str(ROOT / 'cldf/sources.bib'))
    proposed = parse_file(str(PACKAGE / 'proposed-references.bib'))
    assert len(proposed.entries) == 5
    for key, entry in proposed.entries.items():
        assert bibliography.entries[key] == entry
        assert settings['sources'][key]['reference']['editor'] == 'OpenAI Codex'
        assert settings['sources'][key]['reference']['ocr'] is True
        assert meta.flag(key, 'reference', 'editor') == 'OpenAI Codex'
        assert meta.flag(key, 'reference', 'ocr') is True
        assert meta.flag(key, 'reference', 'etymology_provenance') == ('source' if key == SOURCE else 'none')
        assert meta.source_owner[key] == CANONICAL.stem
        formatted = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
        assert formatted
    receipt = json.loads((PACKAGE / 'installation-receipt.json').read_text())
    assert streamed_hash(ROOT / 'data/form-identities.csv') == receipt['durable_identity_sha256']
