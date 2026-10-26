"""Check the complete routed proposal through the real parser, no installation."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata as ud
import argparse

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
import make_cldf
import source_meta

stem = '20260927-behera-whole-additions'
options = argparse.ArgumentParser()
options.add_argument('--all-additions', action='store_true')
args = options.parse_args()
proposal = HERE / ('integration-proposal/all-additions.csv' if args.all_additions
                   else 'integration-proposal/comparison-rows.csv')
settings = [HERE / 'routing-proposal/20260828-sil-korwa-kodaku.yaml',
            HERE / f'routing-proposal/{stem}.yaml']
inputs = [proposal, *settings]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = {str(p.relative_to(HERE)): sha(p) for p in inputs}
rows = list(csv.reader(proposal.open()))
assert len(rows) == len({r[10] for r in rows}) == (2405 if args.all_additions else 1370)
assert all(len(r) == 15 for r in rows)
source_meta.yaml_files = lambda: settings
source_meta.load.cache_clear()
fields = ('lang', 'param', 'old_form', 'gloss', 'native', 'ipa', 'notes',
          'source', 'cognateset', 'etymology', 'entry_key', 'variant_of_key',
          'borrowed_from_key', 'derivation_parent_keys', 'tags')
fixture = HERE / 'integration-proposal/parser-fixture/data/other/forms'
fixture.mkdir(parents=True, exist_ok=True)
checks = {}
for normalization in ('NFC', 'NFD'):
    normalized = [[ud.normalize(normalization, v) if i == 2 else v
                   for i, v in enumerate(r)] for r in rows]
    path = fixture / f'{stem}.csv'
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(normalized)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(path), errors, name=stem)
    assert not errors.getvalue(), errors.getvalue()
    assert len(parsed) == len(rows), (len(parsed), len(rows))
    assert stats == {'converted': 0, 'for_conversion': 0}, stats
    for actual, expected in zip(parsed, normalized):
        assert [getattr(actual, f) for f in fields] == expected, actual.entry_key
        assert ud.normalize('NFC', actual.form) == ud.normalize('NFC', expected[2]), actual.entry_key
        assert actual.is_lone
        assert not any(ud.category(c) == 'Co' or c == '\ufffd' for c in actual.form)
    # The pipeline's first deduper otherwise collapses same-language forms
    # across source prompts, glosses and repeated narrative occurrences.
    meta = source_meta.load()
    dedupe_keys = [(r.lang, r.param, r.form, make_cldf.source_entry_dedupe_key(r, meta))
                   for r in parsed]
    assert all(make_cldf.source_entry_dedupe_key(r, meta) == r.entry_key for r in parsed)
    assert len(set(dedupe_keys)) == len(parsed)
    checks[normalization] = {'rows': len(parsed), 'all_15_fields_exact': True,
                             'ordered_keys_preserved': True, 'unlinked_rows_retained': True,
                             'unwanted_splits': 0, 'errors': 0,
                             'initial_dedupe_distinct_source_records': len(set(dedupe_keys))}
assert pins == {str(p.relative_to(HERE)): sha(p) for p in inputs}, 'Input changed during validation'
report = {'status': 'All proposed rows pass isolated actual parser',
          'inputs': pins, 'parser_sha256': sha(DATA / 'make_cldf.py'),
          'checks': checks, 'canonical_changes': False,
          'not_proven': ['Final literal review', 'Language/dialect registry installation',
                        'Whole-source assembly', 'Full compiled identity/graph behavior']}
(HERE / ('integration-proposal/all-additions-parser-check.json' if args.all_additions
         else 'integration-proposal/comparison-parser-check.json')).write_text(
    json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
