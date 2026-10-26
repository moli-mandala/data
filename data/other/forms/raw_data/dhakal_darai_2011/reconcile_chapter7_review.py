"""Apply original-page decisions without losing candidate evidence or source spelling."""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    review = json.loads((HERE / 'chapter7-original-review-20260927.json').read_text())
    assert [p['printed_page'] for p in review['reviewed_pages']] == list(range(206, 222))
    rows = []
    for name, digest in review['hashes'].items():
        path = HERE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    keyed = {row['source_unit_key']: row for row in rows}
    assert len(rows) == len(keyed) == 441
    skipped = set()
    for decision in review['decisions']:
        keys = decision.get('merge_keys', [decision['key']])
        chosen = [keyed[key] for key in keys]
        row = chosen[0]
        row.setdefault('review_input', copy.deepcopy(chosen))
        row.setdefault('review_decisions', []).append(decision)
        if 'form' in decision:
            row['form'] = unicodedata.normalize('NFC', decision['form'])
        if 'printed_gloss' in decision:
            field = 'candidate_gloss' if ':word:' in row['source_unit_key'] else 'gloss'
            row[field] = decision['printed_gloss']
        for field in ('disposition', 'source_reference', 'source_senses', 'source_interpretation_uncertainty', 'encoding_uncertainty', 'proposed_phonemic'):
            if field in decision:
                row[field] = decision[field]
        if len(keys) > 1:
            row['source_candidate_keys'] = keys
            skipped.update(keys[1:])
    result = []
    for row in rows:
        if row['source_unit_key'] in skipped:
            continue
        row['status'] = 'Owner original-page comparison complete; independent review and metadata integration pending'
        row['review_report'] = 'chapter7-original-review-20260927.json'
        result.append(row)
    represented = {key for row in result for key in row.get('source_candidate_keys', [row['source_unit_key']])}
    assert represented == set(keyed)
    assert len(result) == 441
    assert sum(bool(row.get('encoding_uncertainty')) for row in result) == 1
    return result


if __name__ == '__main__':
    rows = build()
    path = HERE / 'chapter7-reviewed-records.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(len(rows), 'review groups; all 441 raw candidate keys represented;', hashlib.sha256(path.read_bytes()).hexdigest())
