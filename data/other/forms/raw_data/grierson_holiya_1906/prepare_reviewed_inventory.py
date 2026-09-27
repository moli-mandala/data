"""Apply explicit keyed reviews without altering immutable first readings.
This stages an inventory only; it never writes canonical source files.
"""
from pathlib import Path
import json

P = Path(__file__).resolve().parent

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def main():
    reports = sorted(P.glob('independent-*-review-20260926.json')) + sorted(P.glob('owner-*-review-20260926.json'))
    review_data = [(p.name, json.loads(p.read_text())) for p in reports]
    changes = []
    for kind in ('prose', 'specimens'):
        rows = read_jsonl(P / f'{kind}-first-reading.jsonl')
        by_key = {r['source_unit_key']: r for r in rows}
        assert len(rows) == len(by_key)
        for filename, report in review_data:
            for key in report.get('coverage_source_unit_keys', []):
                if key in by_key:
                    by_key[key].setdefault('review_evidence', []).append(filename)
                    by_key[key]['review'] = 'original-image rereading reconciled; see named review evidence; final output audit pending'
            for correction in report.get('corrections', []):
                key = correction['source_unit_key']
                if key not in by_key:
                    continue
                row = by_key[key]
                field = correction['field']
                assert row[field] in (correction['before'], correction['after']), (key, field, row[field])
                row[field] = correction['after']
                row.setdefault('transcription_corrections', []).append({**correction, 'review_file': filename})
                changes.append({**correction, 'review_file': filename})
            for uncertainty in report.get('typed_uncertainties', []):
                if uncertainty['source_unit_key'] in by_key:
                    by_key[uncertainty['source_unit_key']].setdefault('typed_uncertainties', []).append({**uncertainty, 'review_file': filename})
            for refinement in report.get('provenance_refinement', []):
                if refinement['source_unit_key'] in by_key:
                    row = by_key[refinement['source_unit_key']]
                    row['provenance_refinement'] = refinement['note']
                    if row['source_unit_key'].endswith(':came-double-standard'):
                        row['printed_pages'] = [386, 387]
            for decision in report.get('structural_decisions', []):
                keys = decision['keys']
                if all(key in by_key for key in keys):
                    for key in keys:
                        by_key[key].setdefault('structural_review_decisions', []).append(decision)
                        if decision['decision'].startswith('Join physical'):
                            by_key[key]['source_group_keys'] = keys
                            by_key[key]['grouping_decision'] = decision['decision']
        (P / f'{kind}-reviewed-staged.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))
    (P / 'review-application-report.json').write_text(json.dumps({'stage': 'partial review reconciliation; not final proposal', 'corrections_applied': changes, 'reports': [name for name, _ in review_data]}, ensure_ascii=False, indent=2)+'\n')
    print(f'Applied {len(changes)} explicit corrections; first readings unchanged.')

if __name__ == '__main__':
    main()
