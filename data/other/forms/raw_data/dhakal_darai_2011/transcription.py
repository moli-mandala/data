"""Produce a review preview, retaining the source beside house transcription."""
import csv
import json
import unicodedata
from pathlib import Path

from segments.tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent
PROFILE = HERE.parents[4] / 'conversion/dhakal-darai.txt'

# Printed medial-cluster examples (pp.58, 61) contain spaces inside the
# cited word. Normalize only these two occurrences in the house form;
# keep their literal spacing in Original. Reduplicated p.58 remains spaced.
SPACING = {
    'dhakal2011:p58:g806': ('kap. ṭi.ke', 'kap.ṭi.ke'),
    'dhakal2011:p61:g177': ('ɡen ṭʰa', 'ɡenṭʰa'),
}


def build():
    tokenizer = Tokenizer(str(PROFILE))
    proposal = json.loads((HERE / 'proposal.json').read_text())
    output = []
    for row in proposal['candidates']:
        original = row['form']
        text = original
        if row['key'] in SPACING:
            before, text = SPACING[row['key']]
            assert original == before
        form = tokenizer(text, column='IPA').replace(' ', '').replace('#', ' ')
        form = unicodedata.normalize('NFC', form)
        assert form and '�' not in form, row['key']
        output.append({
            'Source_Key': row['key'], 'Language_ID': 'Darai',
            'Form': form, 'Original': original, 'Meaning': row['gloss'],
            'Source_Gloss': row['source_gloss'], 'Page': row['printed_page'],
            'Tags': ';'.join(row['tags']),
            'Variant_Of_Key': row.get('variant_of_key', ''),
            'Source_Analysis': row.get('source_analysis', ''),
        })
    return output


if __name__ == '__main__':
    rows = build()
    with (HERE / 'transcription-preview.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} transcription previews; not installed')
