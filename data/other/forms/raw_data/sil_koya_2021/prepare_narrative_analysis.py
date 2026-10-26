"""Separate reviewed target texts, translations, and test answers; no installation."""
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / 'whole-source-recovery'


def analyze():
    records = [json.loads(line) for line in (HERE / 'narrative-native-first-reading.jsonl').read_text().splitlines()]
    review = json.loads((HERE / 'original-page-review-20260927.json').read_text())
    pages = review['reviewed_pages'] + review['additional_reviewed_pages']
    assert {p['pdf_page'] for p in pages} == set(range(22, 37))
    assert sum(p['units'] for p in pages) == len(records) == 205
    result = []
    for record in records:
        gloss = record['translation_raw']
        answer = None
        answer_notation = None
        if record['kind'] == 'test-question':
            match = re.fullmatch(r'(.*?)\s*(?:<([^<>]*)>|\(([^()]*)\))\s*\.?', gloss)
            assert match, record['source_unit_key']
            gloss = match[1].strip()
            answer = match[2] if match[2] is not None else match[3]
            answer_notation = 'angle brackets' if match[2] is not None else 'parentheses'
            assert gloss.endswith('?'), (record['source_unit_key'], gloss)
        original = record['form_without_question_label']
        editorial = ['[stutter]'] if '[stutter]' in original else []
        display = re.sub(r'\s+', ' ', original.replace('[stutter]', '')).strip()
        result.append({
            **record,
            'original_linguistic_text': original,
            'display_before_transcription_profile': display,
            'gloss': gloss,
            'expected_test_answer': answer,
            'expected_answer_notation': answer_notation,
            'editorial_annotations': editorial,
            'notes': 'Source pairs this printed text segment with a free English translation; no word-level alignment is asserted.' + (' Source marks a stutter at the position retained in Original.' if editorial else ''),
            'status': 'all original pages reviewed; answer separation prepared; metadata/profile and independent output audit pending',
        })
    assert sum(r['expected_test_answer'] is not None for r in result) == 74
    return result


if __name__ == '__main__':
    result = analyze()
    (HERE / 'narrative-reviewed-analysis.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    summary = {
        'status': 'review proposal only; canonical unchanged',
        'units': len(result),
        'questions': sum(r['expected_test_answer'] is not None for r in result),
        'answer_notation': dict(collections.Counter(r['expected_answer_notation'] for r in result if r['expected_test_answer'] is not None)),
        'editorial_stutter_units': sum(bool(r['editorial_annotations']) for r in result),
    }
    (HERE / 'narrative-analysis-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
