"""Stage the sixteen omitted chapter-three clauses; never install partial thesis."""
import csv
import hashlib
import json
from pathlib import Path
from import_source import build as historical_build

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parents[1] / '20260921-dhakal-darai.csv'

def build():
    old_rows, audit = historical_build()
    with CANONICAL.open(newline='') as stream:
        canonical = list(csv.reader(stream))
    assert old_rows == canonical, 'Historical reconstruction differs from installed source input'
    reviewed = json.loads((HERE / 'clause-recovery-reviewed.json').read_text())
    by_key = {r['key']: r for r in audit}
    assert len(reviewed) == 16
    additions = []
    for item in reviewed:
        previous = by_key[item['key']]
        assert previous['status'] == 'excluded'
        source_gloss = previous['raw']['gloss_candidate']
        locator = f"p. {item['printed_page']}, example {item['example']}"
        if item['text_id']:
            locator += f", text {item['text_id']}"
        notes = 'Source interlinear analysis: ' + item['interlinear_form'] + ' / ' + item['interlinear_gloss'] + '.'
        row = ['Darai', '', item['form'], source_gloss, '', item['form'], notes,
               f'dhakal2011darai[{locator}]', '', '', item['key'], '', '', '',
               ' '.join([*item['tags'], 'dialect:Darai:darai_chitwan:Chitwan'])]
        previous['historical_exclusion'] = previous.pop('decision')
        previous.update(status='proposed-whole-clause', output=row, recovery_review=item,
                        scope_warning='Chapter-three recovery only; full thesis extraction remains pending.')
        additions.append(row)
    rows = [*old_rows, *additions]
    assert len(rows) == 501 and len(audit) == 502
    assert len({r[10] for r in rows}) == 501
    assert rows[:485] == canonical
    return rows, audit

if __name__ == '__main__':
    rows, audit = build()
    output = HERE / 'clause-recovery-proposal.csv'
    with output.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    evidence = HERE / 'clause-recovery-proposal-audit.jsonl'
    evidence.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    result = dict(status='proposal only; independent review and full-thesis extraction pending',
        rows=len(rows), audit_units=len(audit), recovered_clauses=16, unchanged_existing_rows=485,
        remaining_audit_only=[r['key'] for r in audit if r['output'] is None],
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [CANONICAL, output, evidence, HERE / 'clause-recovery-reviewed.json']})
    (HERE / 'clause-recovery-proposal-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
