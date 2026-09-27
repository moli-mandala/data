"""Stage the complete Bhalesi chapter without modifying canonical data."""
from pathlib import Path
import csv
import json
import unicodedata
from collections import Counter

P = Path(__file__).resolve().parent
DATA = P.parents[4]
SOURCE = 'bailey1908bhalesi'


def nfc(text):
    return unicodedata.normalize('NFC', text)


def generate():
    units = [json.loads(line) for line in (P / 'full-reviewed.jsonl').read_text().splitlines()]
    assert len(units) == 436 and len({u['source_unit_key'] for u in units}) == 436
    assert {u['printed_page'] for u in units} == set(range(68, 76)) | {28, 53, 54, 'iii'}
    rows, audit = [], []
    for u in units:
        assert u['review'] == 'full-second-reading-original-350dpi-20260926'
        key = u['source_unit_key']
        if u['section'] == 'glossary':
            locator = f"p. {u['printed_page']}, {u['column']} column, item {u['printed_item']}"
        else:
            locator = f"p. {u['printed_page']}, {u['section']}, item {key.split(':')[-1]}"
        emitted = []
        if u['scope'] == 'bhalesi':
            for answer, form in enumerate(u['forms'], 1):
                entry_key = key if answer == 1 else f'{key}:answer{answer}'
                notes = []
                note = u['note'].strip()
                # Workflow notes remain in the audit; source claims remain visible.
                if note and not note.startswith(('Printed abbreviated', 'Table case/', 'Whole source sentence', 'Explicit preceding stem')):
                    notes.append(note)
                if u.get('source_case_label') and not any('Agent case' in n for n in notes):
                    notes.append('Source calls this the Agent case (Ag.).')
                for style in u.get('source_style', []):
                    if style['answer'] == answer:
                        notes.append(f"Source prints {style['span']} in {style['style']}. {style['source_claim']}")
                if u.get('transcription_uncertainty'):
                    uncertainty = u['transcription_uncertainty']
                    notes.append('Transcription uncertainty: ' + uncertainty.get('detail', uncertainty.get('reason', '')))
                tags = list(u['tags'])
                if ' ' in form and 'multiword-expression' not in tags:
                    tags.append('multiword-expression')
                if u['section'] == 'glossary':
                    notes.append('Part of speech follows the English head; the glossary gives no separate POS label.')
                rows.append(['bhal', '', nfc(form), u['gloss'], '', '', ' '.join(notes),
                             f'{SOURCE}[{locator}]', '', '', entry_key,
                             key if answer > 1 else '', '', '', ' '.join(dict.fromkeys(tags))])
                emitted.append(entry_key)
        status = 'ingested' if emitted else ('bound-morphology-audit-only' if u['scope'] == 'bound-morphology' else 'other-lect-control')
        audit.append({**u, 'status': status, 'entry_keys': emitted,
                      'citation_locator': locator, 'language_id': 'bhal' if emitted else '',
                      'transcription_layer': 'Literal source Roman notation with explicit typography; not inferred IPA.'})
    assert len({r[10] for r in rows}) == len(rows)
    legacy_path = P / 'legacy-pilot/20260925-bailey-bhalesi.csv'
    if not legacy_path.exists():
        legacy_path = DATA / 'data/other/forms/20260925-bailey-bhalesi.csv'
    legacy = list(csv.reader(legacy_path.open()))
    assert {r[10] for r in legacy}.issubset({r[10] for r in rows})
    return rows, audit


def main():
    rows, audit = generate()
    with (P / 'proposal.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (P / 'proposal-audit.jsonl').write_text(''.join(json.dumps(u, ensure_ascii=False) + '\n' for u in audit))
    graphemes = set()
    for row in rows:
        clusters = []
        for char in row[2]:
            if unicodedata.combining(char) and clusters:
                clusters[-1] += char
            else:
                clusters.append(char)
        graphemes.update(clusters)
    (P / 'proposal-profile.txt').write_text('Grapheme\tIPA\n' + ''.join(
        f'{char}\t' + ('#' if char == ' ' else '' if char in '.?' else char.replace('ṅ', 'ŋ')) + '\n'
        for char in sorted(graphemes)))
    print(len(rows), 'rows;', len(audit), 'units;', dict(Counter(u['status'] for u in audit)))


if __name__ == '__main__':
    main()
