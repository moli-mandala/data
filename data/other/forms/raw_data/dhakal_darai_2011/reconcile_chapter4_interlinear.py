"""Apply the complete owner page review without changing first readings."""
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    source = HERE / 'chapter4-interlinear-adjudicated-candidates.jsonl'
    review = json.loads((HERE / 'chapter4-interlinear-complete-review-20260927.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == review['input_sha256']
    assert [p['printed_page'] for p in review['reviewed_pages']] == list(range(77, 125))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    by_key = {r['source_unit_key']: r for r in rows}
    skipped = set()
    for decision in review['decisions']:
        keys = decision.get('keys', [decision.get('key')])
        chosen = [by_key[k] for k in keys]
        row = chosen[0]
        row['complete_review_input'] = json.loads(json.dumps(chosen))
        row['complete_review_decision'] = decision
        row['form'] = unicodedata.normalize('NFC', decision.get('literal_decoded_form', decision.get('form', row['form'])))
        if 'printed_gloss' in decision:
            row['candidate_gloss'] = decision['printed_gloss']
        if len(keys) > 1:
            row['source_candidate_keys'] = keys
            row['uncertainty'] = decision['decision']
            row['needs_manual_alignment'] = True
            skipped.update(keys[1:])
        if any(s in decision['decision'] for s in ('conflict', 'contradictory', 'unresolved')):
            row['source_interpretation_uncertainty'] = decision['decision']
    result = []
    for row in rows:
        if row['source_unit_key'] in skipped:
            continue
        row['status'] = 'Owner original-page literal and alignment review complete; independent review pending'
        row['review_status'] = row['status']
        row['review_report'] = 'chapter4-interlinear-complete-review-20260927.json'
        result.append(row)
    represented = {k for r in result for k in r.get('source_candidate_keys', [r['source_unit_key']])}
    represented.update(r['source_unit_key'] for r in result)
    assert set(by_key) <= represented
    assert len(result) == 704
    return result


if __name__ == '__main__':
    rows = build()
    output = HERE / 'chapter4-interlinear-reviewed.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(len(rows), 'reviewed groups', hashlib.sha256(output.read_bytes()).hexdigest())
