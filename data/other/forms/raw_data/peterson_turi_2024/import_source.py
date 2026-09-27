"""Build Turi rows and a complete audit from the pinned, non-OCR PDF snapshot."""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMS = HERE.parent.parent
STEM = '20260921-peterson-turi'
SOURCE = 'peterson2024turi'
DIALECT = 'dialect:Turi:turi_odisha:Odisha'


def build(records):
    rows, audit = [], []
    for record in records:
        raw = record['raw_form']
        key = record['entry_key']
        notes = []
        if record['printed_page'] == 297 and record['printed_item'] == '234.':
            notes.append('Printed 234 in expected slot 254; retain printed locator and page-qualified key.')
        entry = {k: v for k, v in record.items() if k != 'glyphs'}
        entry.update(status='unelicited' if raw == '-' else 'ingested',
                     reason='Explicit source dash' if raw == '-' else '',
                     emitted_keys=[], language='Turi', dialect=DIALECT,
                     review_notes=notes, etymological_links=[])
        if raw != '-':
            # Commas inside annotations (e.g. IA, Magadhan) do not split forms.
            alternatives = re.split(r',\s*(?![^()]*\))', raw)
            for index, part in enumerate(alternatives, 1):
                form = re.sub(r'\s*\([^()]*\)', '', part).strip()
                tags = [DIALECT]
                gloss = record['gloss']
                if gloss == 'you (pl.)':
                    gloss = 'you'
                    tags.append('pl')
                # These are numbered English prompts, not numeric form suffixes.
                gloss = {'jar1': 'jar (1)', 'jar2': 'jar (2)',
                         'plain1': 'plain (1)', 'plain2': 'plain (2)'}.get(gloss, gloss)
                if '̃̃' in form:
                    tags.append('uncertain')
                    notes.append('transcription: redundant nasal mark in the source text layer; '
                                 'retain in Original/Phonemic, collapse only in display profile.')
                if 'likely IA' in raw:
                    tags.append('uncertain')
                    notes.append('borrowing: source explicitly says likely IA; no donor endpoint identified.')
                etymology = ''
                if 'IA' in raw:
                    etymology = (f'Source response: {raw}. '
                                 'The authors use IA for similarity to regional Indo-Aryan forms '
                                 'and likely borrowing, not necessarily ultimate Indo-Aryan origin '
                                 '(Appendix 1 introduction). No specific donor entry is identified.')
                child_key = f'{key}:form{index}'
                citation = f"{SOURCE}[p. {record['printed_page']}, item {record['printed_item'].rstrip('.,')}]"
                row = ['Turi', '', form, gloss, '', form, '', citation, '', etymology,
                       child_key, '', '', '', ' '.join(tags)]
                rows.append([unicodedata.normalize('NFC', value) for value in row])
                entry['emitted_keys'].append(child_key)
        audit.append(entry)
    assert len(audit) == 275
    assert len({r[10] for r in rows}) == len(rows)
    assert all(r[2] and '(' not in r[2] and 'IA' not in r[2] for r in rows)
    return rows, audit


def write(output, rows, audit):
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'{STEM}.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (output / f'{STEM}-audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if not args.output_dir and not args.install:
        parser.error('Pass --output-dir for a proposal or --install for canonical output')
    rows, audit = build(json.loads((HERE / 'records.json').read_text()))
    if args.output_dir:
        write(args.output_dir, rows, audit)
    if args.install:
        with (FORMS / f'{STEM}.csv').open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        (HERE / f'{STEM}-audit.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'source_records': len(audit), 'forms': len(rows),
                      'unelicited': sum(a['status'] == 'unelicited' for a in audit),
                      'linked': 0, 'source_responses_with_IA': sum('IA' in a['raw_form'] for a in audit)}))
