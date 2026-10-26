"""Consolidate fully equivalent occurrences; preserve every source unit in audit.

This writes proposal files only. No canonical installation is implemented here.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

from prepare_whole_reuse_candidates import build as reuse_candidates

from whole_legacy_baseline import legacy_rows, legacy_profile

P = Path(__file__).resolve().parent


def tokens_union(*values):
    return ' '.join(dict.fromkeys(token for value in values for token in value.split()))


def build():
    rows, routes = {}, {}
    for route in ['author-transcription', 'literal-preservation']:
        with (P / ('whole-preview-' + route + '.csv')).open(newline='') as stream:
            for row in csv.reader(stream):
                rows[row[10]], routes[row[10]] = row, route
    legacy = {row[10] for row in legacy_rows()}
    original_rows = {key: row[:] for key, row in rows.items()}
    reuse, reasons = {}, {}
    physical = json.loads((P / 'chapter3-expression-reuse-alignment.json').read_text())['records']
    # These eighteen alignments describe one physical expression recovered
    # twice, not two synonymous or morphologically different attestations.
    for decision in physical:
        key, target = decision['recovered_source_unit_key'], decision['preserved_entry_key']
        assert decision['same_physical_attestation'] and key not in legacy
        source, retained = rows[key], rows[target]
        assert ' '.join(source[2].split()) == ' '.join(retained[2].split())
        assert source[5] == retained[5] and source[8:10] == retained[8:10]
        assert source[11:14] == retained[11:14]
        assert routes[key] == routes[target]
        retained[14] = tokens_union(retained[14], source[14])
        if source[3] != retained[3]:
            retained[6] = (retained[6] + ' Source whole-response translation: ' + source[3]).strip()
        if source[6] and source[6] != retained[6]:
            retained[6] = (retained[6] + ' ' + source[6]).strip()
        reuse[key], reasons[key] = target, 'same physical Chapter 3 attestation; historical key and complete translation retained'
    candidates = reuse_candidates()
    physical_keys = {r['recovered_source_unit_key'] for r in physical} | {r['preserved_entry_key'] for r in physical}
    for group in candidates:
        # Their merged Notes/tags now differ from the occurrence preview.
        if physical_keys.intersection(group['source_unit_keys']):
            continue
        for key, target in group['proposed_reuse'].items():
            assert key not in legacy and key not in reuse
            reuse[key], reasons[key] = target, 'identical exported analysis, Notes, sound route, attribution and source claims'

    def resolve(key):
        seen = set()
        while key in reuse:
            assert key not in seen, key
            seen.add(key)
            key = reuse[key]
        return key

    memberships = collections.defaultdict(list)
    for key in rows:
        memberships[resolve(key)].append(key)
    for target, keys in memberships.items():
        citations = [citation for key in keys for citation in original_rows[key][7].split(';') if citation]
        rows[target][7] = ';'.join(dict.fromkeys(citations))
    output = {name: [] for name in set(routes.values())}
    for key, row in rows.items():
        if key in reuse:
            continue
        for col in [11, 12, 13]:
            if row[col]:
                parents = [resolve(parent) if parent in rows else parent for parent in row[col].split('|')]
                assert key not in parents, (key, parents)
                row[col] = '|'.join(parents)
        output[routes[key]].append(row)
    emitted = {row[10] for group in output.values() for row in group}
    assert legacy <= emitted
    assert len(emitted) + len(reuse) == len(rows)
    audit = []
    for line in (P / 'whole-expanded-metadata-review.jsonl').open():
        record = json.loads(line)
        key = record['source_unit_key']
        if key in rows:
            record['export'] = {
                'entry_key': resolve(key), 'route': routes[key],
                'disposition': 'reuse' if key in reuse else 'emit',
                'reason': reasons.get(key, 'distinct source analysis or retained historical key'),
                'occurrence_citation': original_rows[key][7],
                'occurrence_row': original_rows[key],
            }
            if key not in reuse:
                record['export']['source_unit_keys'] = memberships[key]
        else:
            assert record['disposition'].startswith('audit-only')
            record['export'] = {'disposition': 'audit-only', 'reason': record['disposition']}
        audit.append(record)
    return output, audit


if __name__ == '__main__':
    groups, audit = build()
    files = {}
    for route, rows in groups.items():
        file = P / ('whole-proposed-' + route + '.csv')
        with file.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        files[file.name] = {'rows': len(rows), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    file = P / 'whole-proposed-audit.jsonl'
    with file.open('w') as stream:
        for row in audit:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    files[file.name] = {'rows': len(audit), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    status = {'status': 'Whole-source staging; independent audit and parser validation pending',
              'files': files, 'dispositions': dict(collections.Counter(row['export']['disposition'] for row in audit)),
              'legacy_keys_retained': 485, 'canonical_changed': False,
              'policy': 'Every physical/editorial unit retained in audit, including occurrence citation and pre-reuse row; exact-analysis reuse only, with eighteen reviewed Chapter 3 physical alignments.'}
    (P / 'whole-staging-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))
