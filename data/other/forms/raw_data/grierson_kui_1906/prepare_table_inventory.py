"""Create the complete prompt inventory; never emit or install guessed Kui forms.

The shared LSI prompts bootstrap English labels only. Root compared all blocks
against this volume's original English columns on 2026-09-27. The template's
117 'two daughters' is explicitly overridden by original IV 'To daughters'.
No other source's target form, grammatical analysis or acceptance is imported.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / 'grierson_simla_siraji_1916/full-transcription.tsv'


def prepare():
    census = json.loads((ROOT / 'table-scope-census-20260927.json').read_text())
    prompts = {}
    with TEMPLATE.open(newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if row['stratum'] in ('standard_list', 'translated_sentence') and row['unit'].startswith('item:'):
                number = int(row['unit'].split(':')[1])
                assert number not in prompts
                prompts[number] = row['gloss']
    assert set(prompts) == set(range(1, 242))
    prompts[25] = 'yours'
    prompts[117] = 'to daughters'
    records = []
    for block in census['blocks']:
        for number in range(block['first_prompt'], block['last_prompt'] + 1):
            record = {
                'key': f'grierson1906kui:p{block["target_printed"]}:khondmals-table:prompt{number}',
                'prompt_id': number,
                'english_prompt': prompts[number],
                'printed_page': block['target_printed'],
                'physical_page': block['target_physical'],
                'english_printed_page': block['english_printed'],
                'english_physical_page': block['english_physical'],
                'source_header': 'Kui (Khondmals)',
                'language_id': 'Kui',
                'dialect_proposal': 'grierson1906_kui_khondmals',
                'literal_target_reading': None,
                'status': 'target-transcription-pending',
                'english_review': 'root original-image comparison; case/punctuation normalized for prompt label',
            }
            if number == 117:
                record['template_difference'] = 'Original IV prints To daughters; template label two daughters not reused.'
            if number == 25:
                record['template_difference'] = 'Original IV prints Yours; template label your not reused.'
            if number in (133, 136):
                record.update(status='printed-blank', literal_target_reading=None, source_blank_notation='dotted leaders; independently clarified from sharper alternate scan', lexical_emission=False)
            if 195 <= number <= 200:
                record.update(
                    status='source-cross-reference-pending-target-transcription',
                    physical_record_key='grierson1906kui:p676:khondmals-table:future-bracket195-200',
                    literal_target_reading='As the present tense',
                    reference_prompt_id=number - 16,
                    expansion_policy='Resolve corresponding present form only after original transcription acceptance; preserve future prompt identity and both citations. This is a single printed bracket, not six written target forms.',
                )
            records.append(record)
    assert len(records) == 241 and len({r['key'] for r in records}) == 241
    return records


def main():
    records = prepare()
    output = ROOT / 'table-prompt-inventory.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    report = {
        'status': 'complete prompt inventory; target transcription pending; no canonical installation',
        'prompts': 241,
        'physical_target_units': 236,
        'physical_count_explanation': '233 literal cells +2 printed blanks +1 bracket spanning6 prompts; the prompt inventory is not a physical-token count.',
        'printed_blanks': [133, 136],
        'cross_reference_prompts': list(range(195, 201)),
        'ordinary_target_cells_pending': 233,
        'template': str(TEMPLATE.relative_to(ROOT.parent)),
        'template_sha256': hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'original_english_review': ['649', '650 (equivalent second block)', '657', '661', '665', '669', '673', '677', '681'],
        'review_limitation': 'English label alignment and table structure only; no literal Kui form certification.',
    }
    (ROOT / 'table-prompt-inventory-status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
