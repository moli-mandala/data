"""Snapshot the complete Kurichiya chapter from the pinned local Nair PDF."""
import hashlib
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[6] / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
EXPECTED = '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
doc = fitz.open(PDF)
assert len(doc) == 58
pages = (34, 35, 36)
records = []
for p in pages:
    page = doc[p - 1]
    text = page.get_text(sort=True)
    (HERE / f'pdf{p:02d}-native.txt').write_text(text)
    page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False).save(
        HERE / f'pdf{p:02d}-original.png')
    records.append({'pdf_page': p, 'text_length': len(text),
                    'line_count': len(text.splitlines()),
                    'sha256': hashlib.sha256(text.encode()).hexdigest()})
(HERE / 'snapshot.json').write_text(json.dumps({
    'pdf_sha256': EXPECTED, 'pdf_pages': len(doc), 'pages': records}, indent=2) + '\n')
print(json.dumps(records, indent=2))
