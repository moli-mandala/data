"""Recover all section 3.1.4 comparative forms for review, without installation."""
import hashlib
import json
import re
from pathlib import Path

import fitz

PACKAGE = Path(__file__).resolve().parent
PDF = PACKAGE.parents[5] / 'tmp/pdfs/koya/JLSR2021_029.pdf'
EXPECTED = 'a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d'


def extract():
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
    records = []
    with fitz.open(PDF) as document:
        for pdf_page, expected_counts in [(11, {'salt': 5}),
                                          (12, {'six': 5, 'different': 4, 'village': 4})]:
            text = document[pdf_page - 1].get_text()
            blocks = re.split(r'Words with the gloss ‘([^’]+)’:', text)
            seen = {}
            for offset in range(1, len(blocks), 2):
                gloss, body = blocks[offset:offset + 2]
                matches = list(re.finditer(
                    r'(Utnoor Gondi|Bhamani Gondi|Bhamani Madia|Chintoor Koya|Podia Koya|Malakanagiri Koya):\s*/([^/]+?)/', body))
                seen[gloss] = len(matches)
                for number, match in enumerate(matches, 1):
                    lect, form = match.groups()
                    records.append({
                        'source_unit_key': f'devagnanavaram2021:whole:3.1.4:p{pdf_page-5}:{gloss}:{number}',
                        'pdf_page': pdf_page, 'printed_page': pdf_page - 5,
                        'lect': lect, 'source_form': form, 'gloss': gloss,
                        'raw_source': match[0],
                        'review': 'root compared every cell with original PDF pages11–12',
                        'interpretation': 'comparative attestation; no cognate edge inferred',
                        'transcription_note': 'Section3.1.4 explicitly identifies G as a voiced velar fricative and the corresponding r as an alveolar flap; source slash delimiters are notation.',
                        'status': 'first reading; source metadata, conversion and independent output audit pending',
                    })
            assert seen == expected_counts, seen
    assert len(records) == 18
    return records


if __name__ == '__main__':
    records = extract()
    output = PACKAGE / 'whole-source-recovery/comparative-first-reading.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(f'{len(records)} comparative attestations; no canonical changes')
