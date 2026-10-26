"""Locate source-drawn low lines by vector geometry, without glyph inference."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[6] / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
EXPECTED = '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
doc = fitz.open(PDF)
results = []
for number in (34, 35, 36):
    page = doc[number - 1]
    lines = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            chars = [c for span in line['spans'] for c in span['chars']]
            if chars:
                lines.append(chars)
    for drawing in page.get_drawings():
        rect = drawing['rect']
        if not (rect.width < 8 and rect.y0 < 720 and rect.height < 1):
            continue
        cx = (rect.x0 + rect.x1) / 2
        candidates = []
        for chars in lines:
            for i, char in enumerate(chars):
                b = fitz.Rect(char['bbox'])
                if b.x0 - 0.8 <= cx <= b.x1 + 0.8 and abs(rect.y0 - b.y1) <= 12:
                    candidates.append((round(abs(rect.y0 - b.y1), 2), char['c'],
                                       ''.join(c['c'] for c in chars), i, [round(x, 2) for x in b]))
        candidates.sort()
        results.append({'pdf_page': number,
                        'rect': [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
                        'candidate_characters': candidates[:3]})
assert len(results) == 12
(HERE / 'underline-vector-census.json').write_text(json.dumps({
    'source_pdf_sha256': EXPECTED, 'status': 'geometry census; visual review required',
    'drawn_low_lines': len(results), 'records': results}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'drawn_low_lines': len(results), 'identified': sum(bool(r['candidate_characters']) for r in results)}))
