"""Installed whole-source Haijong checks, scoped without a database build."""
import csv
import hashlib
import importlib.util
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_haijong_1903'
CSV = DATA / 'data/other/forms/20260925-grierson-haijong.csv'
PROFILE = DATA / 'conversion/grierson-haijong-1903.txt'
SOURCE = 'grierson1903haijong'
LEGACY_PROMPTS = [1,2,3,4,6,7,8,9,10,11,13,14,20,23]


def installed():
    with CSV.open(encoding='utf-8', newline='') as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE/'audit.jsonl').read_text().splitlines()]


def test_whole_source_installation_matches_regeneration_and_frozen_review():
    spec = importlib.util.spec_from_file_location('haijong_installed_full', PACKAGE/'import_source.py')
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(rows) == 896 and len(audit) == 907
    assert len({r[10] for r in rows}) == 896
    assert sum(bool(r[4]) for r in rows) == 385
    report = json.loads((PACKAGE/'independent-final-output-audit-20260926.json').read_text())
    assert report['status'] == 'pass' and report['material_errors'] == 0
    for canonical, reviewed in [(CSV,'full-proposed.csv'), (PACKAGE/'audit.jsonl','full-proposed-audit.jsonl'), (PROFILE,'full-profile-proposed.txt')]:
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == report['input_sha256'][reviewed]
    assert Counter(a['status'] for a in audit) == {
        'staged':857, 'parallel_native_witness':26, 'bound_marker':12,
        'source_blank':11, 'comparison_control':1,
    }


def test_legacy_source_identity_survives_corrected_readings():
    import assign_form_ids
    rows = {r[10]:r for r in installed()}
    with (PACKAGE/'transcription.tsv').open() as handle:
        historical = {int(r['prompt']):r for r in csv.DictReader(handle,delimiter='\t')}
    for prompt in LEGACY_PROMPTS:
        key = f'{SOURCE}:p354:haijong:prompt:{prompt}'
        assert key in rows
        old = {'Original':historical[prompt]['printed_form_review'], 'Gloss':historical[prompt]['gloss']}
        new = {'Original':rows[key][2], 'Gloss':rows[key][3], 'Native':rows[key][4]}
        # Durable ID reconciliation uses the unchanged source key, not corrected
        # transcription, Native, profile output or newly attached source notes.
        assert assign_form_ids.fingerprint(old, key) == assign_form_ids.fingerprint(new, key)
    assert rows[f'{SOURCE}:p354:haijong:prompt:1'][2:4] == ['Ăk','one']
    assert rows[f'{SOURCE}:p354:haijong:prompt:20'][2:4] == ['Tay','thou']
    assert rows[f'{SOURCE}:p354:haijong:prompt:23'][2:4] == ['Tay','you']


def test_metadata_dialects_and_preservation_contract():
    import source_meta
    metadata = yaml.safe_load(CSV.with_suffix('.yaml').read_text())
    assert metadata['defaults']['transcription']['profile'] == 'grierson-haijong-1903'
    assert metadata['defaults']['transcription']['preserve_hyphens'] is True
    assert metadata['sources'][SOURCE]['reference']['ocr'] is True
    assert metadata['sources'][SOURCE]['forms']['split_alternates'] is False
    assert metadata['sources'][SOURCE]['identity']['dedupe_by_entry_key'] is True
    assert source_meta.SourceMeta().transcription(SOURCE,CSV,'Hajong')[0] == 'grierson-haijong-1903'
    with (DATA/'cldf/dialects.csv').open() as handle:
        registry = {r['Tag']:r for r in csv.DictReader(handle)}
    tags = {tag for r in installed() for tag in r[14].split() if tag.startswith('dialect:')}
    assert len(tags) == 2
    for tag in tags:
        assert registry[tag]['Language_ID'] == 'Hajong'
        assert registry[tag]['Latitude'] == registry[tag]['Longitude'] == ''
    assert (DATA/'cldf/sources.bib').read_text().count('@book{grierson1903haijong,') == 1


def test_scoped_parser_roundtrip_original_native_graph_tags_and_citations():
    import make_cldf
    import profile_policy
    tokenizer = Tokenizer(str(PROFILE))
    raw = {r[10]:r for r in installed()}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats['converted'] == 896
    assert {r.entry_key for r in parsed} == set(raw)
    assert len({r.id for r in parsed}) == 896
    assert 'grierson-haijong-1903' not in profile_policy.audit({})
    for row in parsed:
        source = raw[row.entry_key]
        assert row.old_form == source[2]
        assert row.native == source[4]
        assert row.ipa == source[5] == ''
        assert row.notes == source[6]
        assert row.source == source[7]
        assert row.tags == source[14]
        assert row.variant_of_key == source[11]
        assert not row.variant_of_key or row.variant_of_key in raw
        expected = unicodedata.normalize('NFC',tokenizer(source[2],column='IPA').replace(' ','').replace('#',' '))
        assert row.form == expected and '�' not in row.form
        if row.native:
            assert f'{SOURCE}[p. 216, native line ' in row.source
    assert sum(bool(r.native) for r in parsed) == 385
    assert sum('uncertain' in r.tags.split() for r in parsed) == 4
