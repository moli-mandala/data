"""Conservative exact-analysis reuse candidates, never an installation decision."""
import collections
import csv
import hashlib
import json
from pathlib import Path

from whole_legacy_baseline import legacy_rows, legacy_profile

P = Path(__file__).resolve().parent
PROVENANCE_ONLY = {'text_id', 'text_ids', 'source_reference'}


def fingerprint(row, exported=None, route=None):
    claims = {k: v for k, v in row['source_claims'].items() if k not in PROVENANCE_ONLY}
    # Do not merge differing source analyses, grammatical roles, uncertainty,
    # locality or explanations just because form and short gloss happen to match.
    return json.dumps({
        'form': row['form'], 'gloss': row['proposed_grammar']['gloss'],
        'tags': sorted(set(row['proposed_grammar']['tags']) |
                       set(row['proposed_grammar'].get('existing_tags', []))), 'claims': claims,
        'roles': row.get('source_agreement_roles'),
        'attribution': row.get('source_attribution'),
        'notation': row.get('source_notation_decision'),
        'phonemic_policy': row.get('phonemic_policy'),
        'conflicting_tags': row.get('conflicting_tag_decision'),
        'open_review': row['open_review'],
        # The actual export is authoritative for retained Notes, legacy tags,
        # sound routing and graph interpretations. Citation and entry key alone
        # may differ between genuinely equivalent occurrences.
        'exported_analysis': ([sorted(set(value.split())) if i == 14 else value
                              for i, value in enumerate(exported) if i not in {7, 10}]
                             if exported is not None else None),
        'export_route': route,
    }, ensure_ascii=False, sort_keys=True)


def build():
    exports = {}
    for route in ['author-transcription', 'literal-preservation']:
        with (P / ('whole-preview-' + route + '.csv')).open(newline='') as stream:
            for row in csv.reader(stream):
                assert row[10] not in exports
                exports[row[10]] = (row, route)
    groups = collections.defaultdict(list)
    for line in (P / 'whole-expanded-metadata-review.jsonl').open():
        row = json.loads(line)
        if row['disposition'].startswith('audit-only'):
            continue
        exported, route = exports[row['source_unit_key']]
        groups[fingerprint(row, exported, route)].append(row)
    installed_keys={row[10] for row in legacy_rows()}
    output=[]
    for key, rows in groups.items():
        if len(rows)<2:
            continue
        retained=[r['source_unit_key'] for r in rows if r['source_unit_key'] in installed_keys]
        target=retained[0] if retained else rows[0]['source_unit_key']
        retained=retained or [target]
        output.append({
        'form': rows[0]['form'], 'gloss': rows[0]['proposed_grammar']['gloss'],
        'analysis_fingerprint_sha256': hashlib.sha256(key.encode()).hexdigest(),
        'source_unit_keys': [r['source_unit_key'] for r in rows],
        'printed_pages': [r['printed_page'] for r in rows],
        'proposed_retained_keys':retained,
        'proposed_reuse':{r['source_unit_key']:target for r in rows if r['source_unit_key'] not in retained},
        'occurrence_context':[{'source_unit_key':r['source_unit_key'],
            'evidence_file':r['evidence_file'],'evidence_line':r['evidence_line'],
            'printed_page':r['printed_page'],'parent_expression_key':r.get('parent_expression_key'),
            'source_text_id':r['source_claims'].get('text_id'),
            'source_text_ids':r['source_claims'].get('text_ids')} for r in rows],
        'decision': 'Exact-analysis candidate only; export must preserve every occurrence citation and source key in audit, and every installed key as a row',
        })
    return output


if __name__ == '__main__':
    groups = build()
    report = {
        'status': 'Review candidates only; no row dropped or key reassigned',
        'metadata_sha256': hashlib.sha256((P / 'whole-expanded-metadata-review.jsonl').read_bytes()).hexdigest(),
        'groups': groups, 'group_count': len(groups),
        'candidate_repeated_occurrences': sum(len(r['source_unit_keys']) - 1 for r in groups),
        'legacy_key_policy': 'Every installed source key must survive; never consolidate two existing keys',
    }
    (P / 'whole-exact-reuse-candidates.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'groups'}))
