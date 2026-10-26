"""Join reviewed table readings to the complete inventory without installation.

Every prompt remains visible, including unfinished transcription and source blanks.
Independent caveats are carried into the record, never erased by an agreement.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

from prepare_table_inventory import prepare

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]


def main():
    raw = json.loads((HERE / 'table-first-reading.json').read_text())
    readings = {r['prompt_id']: r for r in raw['readings']}
    assert len(readings) == len(raw['readings'])
    review_path = HERE / 'independent-table-first25-review-20260927.json'
    review = json.loads(review_path.read_text())
    independent = {r['prompt_id']: r for r in review['items']}
    witnesses = {r['archive_leaf']: r for r in json.loads((HERE / 'alternate-table-acquisition.json').read_text())}
    for witness in witnesses.values():
        original = WORKSPACE / witness['path']
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == witness['sha256']
    records = prepare()
    for record in records:
        prompt = record['prompt_id']
        if prompt not in readings:
            continue
        reading = readings[prompt]
        witness = witnesses[reading.get('source_leaf', 669)]
        assert witness['printed_page'] == record['printed_page']
        record['alternate_original'] = witness
        record['first_reading'] = reading
        record['status'] = 'first-reading-awaiting-independent-review'
        for form in reading['forms']:
            assert form and '\ufffd' not in form
            assert unicodedata.normalize('NFC', form) == form, (prompt, form)
        if prompt in independent:
            decision = independent[prompt]
            assert reading['literal'] == decision['old_literal'], prompt
            assert reading['forms'] == decision['forms'], prompt
            record['independent_review'] = decision
            record['status'] = 'independently-reviewed-reading; full-table-gates-pending'
            # Preserve every original uncertainty and its adjudication history.
            if prompt in (10, 11, 12):
                record['uncertainty_adjudication'] = {
                    'status': 'first-reading question resolved by independent original review',
                    'reason': 'Repeated acute stroke supports ś in10/12; distinct final bar supports ō in11.',
                }
            if prompt in (15, 18, 21, 24, 22, 25):
                record['retained_uncertainty'] = {
                    'type': 'transcription',
                    'reason': decision['evidence'],
                    'policy': 'Provisional literal Unicode; no phonemic interpretation or harmonization.',
                }
        # This artifact records candidate forms, not approved canonical emissions.
        record['canonical_emission_authorized'] = False
    output = HERE / 'table-reading-audit.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    report = {
        'status': ('whole inventory with all written cells first-read; independent review pending; no canonical source installed'
                   if not any(r['status'] == 'target-transcription-pending' for r in records)
                   else 'whole inventory with partial first reading; no canonical source installed'),
        'prompt_records': len(records),
        'first_reading_prompts': len(readings),
        'independently_reviewed_prompts': len(independent),
        'source_blanks': [133, 136],
        'future_cross_references': list(range(195, 201)),
        'ordinary_cells_awaiting_first_reading': sum(r['status'] == 'target-transcription-pending' for r in records),
        'retained_independent_typography_caveats': [15, 18, 21, 22, 24, 25],
        'candidate_headless_cell': [r['prompt_id'] for r in readings.values() if not r['forms']],
        'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
        'audit_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'scope': 'Table only. Whole chapter, all specimens, outside-chapter comparisons, metadata and independent whole-output review remain separate gates.',
    }
    (HERE / 'table-reading-status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
