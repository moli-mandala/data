"""Close all historical, text and outside witness records against the proposal."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
OUT = HERE / 'integration-proposal'
paths = {
    'historical': DATA / 'data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv',
    'legacy_repair': HERE / 'legacy-review/legacy-metadata-repair.jsonl',
    'legacy_rows': HERE / 'legacy-review/legacy-repaired-candidate.csv',
    'comparison': HERE / 'comparisons/comparison-review-accounting.jsonl',
    'additions_audit': OUT / 'all-additions-audit.jsonl',
    'additions_rows': OUT / 'all-additions.csv',
    'text_events': HERE / 'interlinear/tagged-blocks-v5.json',
    'outside': HERE / 'outside-review/outside-candidate.jsonl',
    'baseline': HERE / 'legacy-review/reconciliation.json',
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = {k: sha(p) for k, p in paths.items()}
jlines = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
historical = list(csv.DictReader(paths['historical'].open()))
repairs = {r['entry_key']: r for r in jlines(paths['legacy_repair'])}
comparisons = {r['historical_audit_row_one_based']: r for r in jlines(paths['comparison'])}
new_audit = jlines(paths['additions_audit'])
new_by_key = {r.get('entry_key', r.get('source_unit_key')): r for r in new_audit}
legacy_rows = list(csv.reader(paths['legacy_rows'].open()))
new_rows = list(csv.reader(paths['additions_rows'].open()))
all_rows = legacy_rows + new_rows
all_keys = {r[10] for r in all_rows}
assert len(all_keys) == len(all_rows)
assert len(historical) == 6135 and len(comparisons) == 1628
ledger = []
for index, record in enumerate(historical, 1):
    emitted = []
    held = []
    if index in comparisons:
        reviewed = comparisons[index]
        assert record == reviewed['historical_audit_record']
        disposition = reviewed['disposition']
        for key in reviewed.get('candidate_keys', []):
            assert key in new_by_key
            if key in all_keys:
                emitted.append(key)
            else:
                held.append(key)
    elif record['Status'] == 'installed':
        disposition = 'legacy-reconciled-proposal'
        for key in record['Entry_Keys'].split('|'):
            assert key in repairs
            emitted.extend(repairs[key]['emitted_keys'])
    else:
        assert record['Status'] == 'missing'
        disposition = 'source-blank'
    superseded = []
    if disposition == 'historical blank superseded by physical response':
        assert record['Site_Code'] == 'V' and record['Item'] == '208'
        superseded = [key for key, candidate in new_by_key.items()
                      if candidate.get('source_code') == 'V' and candidate.get('item') == 208]
        assert len(superseded) == 1 and superseded[0] in all_keys
    ledger.append({'audit_key': f'behera2022korwakodaku:historical-record:{index}',
                   'unit_type': 'historical-wordlist-record', 'raw_record': record,
                   'disposition': disposition, 'proposed_entry_keys': emitted,
                   'held_candidate_keys': held,
                   'superseded_by_candidate_keys': superseded})

physical_to_rows = defaultdict(set)
physical_to_held = defaultdict(set)
held_text_keys = set()
for record in new_audit:
    key = record.get('entry_key', record.get('source_unit_key'))
    is_held_text = record['integration_disposition'].startswith('held-internal-word-alignment')
    if key not in all_keys and not is_held_text:
        continue
    physical_map = physical_to_held if is_held_text else physical_to_rows
    if is_held_text:
        held_text_keys.add(key)
    for field in ('physical_p_keys', 'physical_g_keys', 'physical_f_keys', 'physical_q_keys'):
        for physical in record.get(field, []):
            physical_map[physical].add(key)
    for field in ('physical_p_key', 'physical_g_key', 'physical_answer_key'):
        if record.get(field):
            physical_map[record[field]].add(key)
events = json.loads(paths['text_events'].read_text())
for event in events:
    ledger.append({'audit_key': event['physical_key'], 'unit_type': 'text-marker-block',
                   'raw_record': event, 'proposed_entry_keys': sorted(physical_to_rows[event['physical_key']]),
                   'held_candidate_keys': sorted(physical_to_held[event['physical_key']]),
                   'disposition': 'raw-text-block-preserved; see scoped candidate review'})
assert {key for keys in physical_to_held.values() for key in keys} == held_text_keys
assert len(held_text_keys) == 66
for record in jlines(paths['outside']):
    key = record['entry_key']
    ledger.append({'audit_key': key, 'unit_type': 'outside-witness', 'raw_record': record,
                   'proposed_entry_keys': [key] if key in all_keys else [],
                   'held_candidate_keys': [key] if record['disposition'].startswith('held-') else [],
                   'disposition': record['disposition']})
unknown = json.loads(paths['baseline'].read_text())['undefined_source_codes']
for record in unknown:
    key = f'behera2022korwakodaku:unknown-code:{record["Unknown_Site_Code"]}:item{record["Item"]}'
    ledger.append({'audit_key': key, 'unit_type': 'undefined-wordlist-code', 'raw_record': record,
                   'proposed_entry_keys': [], 'disposition': 'held-undefined-source-code'})
covered = {key for r in ledger for key in r['proposed_entry_keys']}
assert covered == all_keys, {'missing': sorted(all_keys - covered), 'unknown': sorted(covered - all_keys)}
assert len({r['audit_key'] for r in ledger}) == len(ledger)
assert pins == {k: sha(p) for k, p in paths.items()}, 'Inputs changed during reconciliation'
(OUT / 'whole-source-witness-audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ledger))
report = {'status': 'Every proposed entry has source-witness coverage; acceptance pending',
          'inputs': pins, 'audit_records': len(ledger),
          'unit_types': dict(Counter(r['unit_type'] for r in ledger)),
          'legacy_rows': len(legacy_rows), 'addition_rows': len(new_rows),
          'all_proposed_rows': len(all_rows), 'covered_entry_keys': len(covered),
          'held_text_candidates_explicitly_linked': len(held_text_keys),
          'superseded_blank_links': sum(bool(r.get('superseded_by_candidate_keys')) for r in ledger),
          'canonical_changes': False,
          'limits': ['Links prove accounting, not correctness of source interpretation',
                     'Independent corrections and final fresh audit still required',
                     'Compiled/database/browser gates deferred']}
(OUT / 'whole-source-witness-status.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))
