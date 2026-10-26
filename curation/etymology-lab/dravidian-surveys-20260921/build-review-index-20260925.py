"""Summarise registered 2026-09-25 review-only research passes."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
LAB = P.parent
registered = json.loads((P / 'goal-pass-registry.json').read_text())
lines = [
    '# Dravidian survey research review queue — 2026-09-25',
    '',
    '**All entries below are pending proposals, not accepted etymology links. No database or site rebuild was run.**',
    '',
    'Passes with proposals link to numbered, evidence-bearing language batches and canonical dry-run validation. Holds-only passes document why no link was proposed. Counts below refer to the records newly assessed in that pass; earlier proposals for the same source are kept in their own reviews.',
    '',
    '| Survey pass | Proposed records | Assignment rows | Research holds | Review |',
    '|---|---:|---:|---:|---|',
]
totals = dict(passes=0, proposed=0, rows=0, held=0)
for name in registered:
    if not name.endswith('20260925'):
        continue
    folder = LAB / name
    accepted, held = {}, {}
    for file in sorted(folder.glob('proposed-decisions*.json')):
        decisions = json.loads(file.read_text())
        assert decisions['status'] == 'pending-review', file
        accepted.update({row['record']['ID']: row for row in decisions['accepted']})
        held.update({row['record']['ID']: row for row in decisions['held']})
    assert not set(accepted) & set(held), name
    assert (folder / 'REVIEW.md').is_file() and (folder / 'STATE.md').is_file(), name
    if accepted:
        assert list(folder.glob('checkpoint-*-validation.json')), name
    rows = sum(len(row.get('components', [row['parent']])) for row in accepted.values())
    totals['passes'] += 1
    totals['proposed'] += len(accepted)
    totals['rows'] += rows
    totals['held'] += len(held)
    label = name.removesuffix('-20260925').replace('-dravlex', ' · DravLex').replace('-lsi', ' · LSI').replace('-', ' ')
    lines.append(f'| {label} | {len(accepted)} | {rows} | {len(held)} | [Review](../{name}/REVIEW.md) |')
lines += [
    '',
    f"Registered passes: **{totals['passes']}**. Newly assessed records: **{totals['proposed'] + totals['held']:,}** ({totals['proposed']:,} pending proposals, {totals['held']:,} research holds). Proposed assignment rows: **{totals['rows']:,}**.",
    '',
    '**Source-identity exclusion:** [LSI Gadabā](../gadaba-lsi-20260925/STATE.md) has 222 records identified as Bodo Gadaba/Gutob (`bodo1267`, Munda) but imported under Mudhili Gadaba (`mudh1235`, Dravidian). They have zero Dravidian proposals and are excluded from the research counts above; the source mapping itself has not been changed.',
    '',
    'The [all-source inventory snapshot](all-source-priorities-20260925-current.json) ranks unexamined Dravidian source groups at the time it was generated; later registered and active passes may not yet be reflected there. Pending proposals require user review before any `save-pass` action.',
]
(P / 'REVIEW-20260925.md').write_text('\n'.join(lines) + '\n')
print(totals)
