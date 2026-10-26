"""Validate staged settings via SourceMeta and unchanged generic parse_file."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
import source_meta
import make_cldf
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

STEM = '20260927-nair-mannan'
FIELDS = ('lang', 'param', 'old_form', 'gloss', 'native', 'ipa', 'notes', 'source',
    'cognateset', 'etymology', 'entry_key', 'variant_of_key', 'borrowed_from_key', 'derivation_parent_keys', 'tags')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (HERE / 'root-acceptance-draft.json').exists():
        draft = json.loads((HERE / 'root-acceptance-draft.json').read_text())
        assert draft['installation_approved'] is False
    else:
        freeze = json.loads((PACKAGE / 'review-freeze.json').read_text())
        assert sha(PACKAGE / 'review-freeze.json') == '8fe6620df601435704e65f2f66b1382f34dbb507392123b3ee48ad30cea4829a'
        draft = {'files': dict(freeze['files_sha256'])}
        for name in ['installation/20260927-nair-mannan.yaml', 'installation/accepted-audit-proposal.jsonl']:
            draft['files'][name] = sha(PACKAGE / name)
    for name, digest in draft['files'].items():
        assert sha(PACKAGE / name) == digest, name
    rows = list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    assert len(rows) == 154
    registry = {r['ID'] for r in csv.DictReader((DATA / 'cldf/languages.csv').open(newline=''))}
    assert {'Mannan', 'Malayalam'} <= registry
    assert all(r[0] in {'Mannan', 'Malayalam'} and set(r[14].split()) <= GRAMMATICAL_TAGS | GENDER_TAGS for r in rows)
    old_discovery = source_meta.yaml_files
    checks = {}
    try:
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            folder = Path(temp)
            settings = folder / (STEM + '.yaml')
            settings.write_bytes((HERE / (STEM + '.yaml')).read_bytes())
            meta = source_meta.SourceMeta([settings])
            assert len(meta.sources) == 2
            combined_meta = source_meta.SourceMeta([*source_meta.yaml_files(), settings])
            assert set(meta.sources) <= set(combined_meta.sources)
            assert meta.file_flag(STEM + '.csv', 'identity', 'dedupe_by_entry_key') is True
            for key in meta.sources:
                assert meta.flag(key, 'reference', 'ocr') is False
                assert meta.flag(key, 'reference', 'editor') == 'OpenAI Codex'
            source_meta.yaml_files = lambda: [settings]
            source_meta.load.cache_clear()
            for normalization in ('NFC', 'NFD'):
                inputs = [[unicodedata.normalize(normalization, v) if i == 2 else v for i, v in enumerate(r)] for r in rows]
                path = folder / (STEM + '.csv')
                with path.open('w', newline='') as output:
                    csv.writer(output).writerows(inputs)
                errors = io.StringIO()
                parsed, stats = make_cldf.parse_file(str(path), errors, name=STEM)
                assert not errors.getvalue(), errors.getvalue()
                assert len(parsed) == 154 and stats == {'converted': 0, 'for_conversion': 0}
                for actual, expected in zip(parsed, inputs):
                    assert [getattr(actual, field) for field in FIELDS] == expected, actual.entry_key
                    assert unicodedata.normalize('NFC', actual.form) == unicodedata.normalize('NFC', expected[2])
                checks[normalization] = {'rows': 154, 'all15fields_exact': True, 'ordered_keys_exact': True, 'literal_display': True}
    finally:
        source_meta.yaml_files = old_discovery
        source_meta.load.cache_clear()
    for name, digest in draft['files'].items():
        assert sha(PACKAGE / name) == digest, name
    report = dict(status='pass: staged SourceMeta and actual generic parser fixture', checks=checks,
        source_reference_flags=2, input_csv_sha256=sha(PACKAGE / 'proposed.csv'),
        combined_current_registry_and_staged_yaml='all current settings plus staged file validate with unique source declarations',
        staged_settings_sha256=sha(HERE / (STEM + '.yaml')),
        registered_languages='existing Mannan/mann1246 and Malayalam; no new base, lect or coordinates',
        isolation='Only YAML discovery is isolated to the source-local fixture; parse_file and conversion implementations unchanged.',
        canonical_changes=False, canonical_source_meta_and_parser_gate='required after authorized installation')
    (HERE / 'actual-parser-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
