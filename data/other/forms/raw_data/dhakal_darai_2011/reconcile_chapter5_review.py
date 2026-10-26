"""Reconcile owner visual decisions, preserving every candidate and raw glyph."""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    review = json.loads((HERE / 'chapter5-original-review-20260927.json').read_text())
    assert [p['printed_page'] for p in review['reviewed_pages']] == list(range(125, 146))
    inputs = {}
    for name, digest in review['hashes'].items():
        source = HERE / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        inputs[name] = [json.loads(line) for line in source.read_text().splitlines()]
    rows = inputs['chapter5-first-reading.jsonl'] + inputs['chapter5-interlinear-layout-v2-candidates.jsonl']
    keyed = {row['source_unit_key']: row for row in rows}
    assert len(keyed) == len(rows)
    skipped = set()
    for decision in review['decisions']:
        keys = decision.get('keys', [decision.get('key')])
        if keys == [None]:
            continue
        chosen = [keyed[key] for key in keys]
        row = chosen[0]
        row['review_input'] = copy.deepcopy(chosen)
        row.setdefault('review_decisions', []).append(decision)
        if 'form' in decision:
            row['form'] = unicodedata.normalize('NFC', decision['form'])
        if 'printed_gloss' in decision:
            row['candidate_gloss'] = decision['printed_gloss']
        if 'disposition' in decision:
            row['disposition'] = decision['disposition']
        if len(keys) > 1:
            row['source_candidate_keys'] = keys
            skipped.update(keys[1:])
            if 'form' not in decision:
                row['form'] = ''.join(item['form'] for item in chosen)
                row['encoding_uncertainty'] = decision['decision']
                row['proposed_phonemic'] = ''
        if any(word in decision['decision'] for word in ('contradiction', 'mismatch')):
            row['source_interpretation_uncertainty'] = decision['decision']
    surface = keyed['dhakal2011:p141:ch5:surface:l60']
    surface['review_input'] = copy.deepcopy(surface)
    surface['form'] = surface['form'].replace('ə⟦U+F0E2⟧', 'ə̃').replace('ə⟦F0E2⟧', 'ə̃')
    surface['review_decisions'] = [{'decision': 'Original nasal mark belongs over ə in cʰə̃uḍi, as in its aligned token.'}]
    result = []
    for row in rows:
        if row['source_unit_key'] in skipped:
            continue
        row['status'] = 'Owner original-page comparison complete; independent review and metadata integration pending'
        row['review_report'] = 'chapter5-original-review-20260927.json'
        result.append(row)
    represented = {key for row in result for key in row.get('source_candidate_keys', [row['source_unit_key']])}
    assert represented == set(keyed)
    assert len(result) == 541
    assert sum(row.get('disposition') == 'English-gloss-control' for row in result) == 6
    return result


if __name__ == '__main__':
    result = build()
    output = HERE / 'chapter5-reviewed-records.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result))
    print(len(result), 'review groups; all 543 original candidates represented;', hashlib.sha256(output.read_bytes()).hexdigest())
