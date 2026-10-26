"""Reproduce an independent sample from hash-pinned whole-source staging."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent


def source_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'key', 'source_unit_key', 'entry_key', 'Source_Key', 'source_key'} and isinstance(item, str) and item.startswith('dhakal2011:'):
                yield item
            else:
                yield from source_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from source_keys(item)


def sample(seed, count):
    freeze = json.loads((P / 'whole-proposal-freeze.json').read_text())
    dependency_report = P/'whole-dependency-patch-parity-20260927.json'
    changes = {}
    if dependency_report.exists():
        changes = {r['file']: r for r in json.loads(dependency_report.read_text())['changed_pinned_code']}
    for name, expected in freeze['files'].items():
        if name in changes:
            assert expected == changes[name]['reviewed_code_sha256']
            expected = changes[name]['dependency_patched_sha256']
        assert hashlib.sha256((P / name).read_bytes()).hexdigest() == expected, name
    excluded = set()
    exclusion_files = ['chapters12-independent-selection.json', 'preview/acceptance-sample.json']
    exclusion_files += [path.name for path in sorted(P.glob('whole-independent-selection-*.json'))
                        if path.name != f'whole-independent-selection-{seed}.json']
    for name in exclusion_files:
        excluded.update(source_keys(json.loads((P / name).read_text())))
    audits = [json.loads(line) for line in (P / 'whole-proposed-audit.jsonl').open()]
    # An already reviewed occurrence may now point to a consolidated target.
    excluded.update(r['export']['entry_key'] for r in audits
                    if r['source_unit_key'] in excluded and 'entry_key' in r['export'])
    eligible = [r for r in audits if r['export']['disposition'] == 'emit' and r['source_unit_key'] not in excluded]
    chosen = random.Random(seed).sample(sorted(eligible, key=lambda r: r['source_unit_key']), count)
    exported = {}
    for route in ['author-transcription', 'literal-preservation']:
        with (P / ('whole-proposed-' + route + '.csv')).open(newline='') as stream:
            exported.update((row[10], row) for row in csv.reader(stream))
    for row in chosen:
        row['sample_exported_row'] = exported[row['export']['entry_key']]
    return {'seed': seed, 'count': count, 'freeze': freeze,
            'exclusion_files': exclusion_files, 'excluded_keys': sorted(excluded),
            'sampling': 'Uniform random emitted entries, sorted by source key before seeded sampling; targeted edge reviews are additional, not substituted.',
            'records': chosen}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    result = sample(args.seed, args.count)
    file = P / f'whole-independent-selection-{args.seed}.json'
    assert not file.exists(), 'Samples are immutable; use a fresh seed.'
    file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(file)
