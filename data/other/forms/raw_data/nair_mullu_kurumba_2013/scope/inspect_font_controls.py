"""Bounded immutable original detail evidence; no accepted output."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/SOURCE_INGESTION_CHECKLIST.md').exists())
PDF = ROOT / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
doc = fitz.open(PDF)
evidence = []
for page_number, label, clip in [(12, 'outside-classification', (80, 75, 530, 450)),
    (55, 'noun-dots-lowline', (85, 420, 530, 635)),
    (56, 'pronoun-lowline', (90, 485, 430, 695)),
    (57, 'participle-tilde-dots', (85, 465, 420, 610))]:
    page = doc[page_number - 1]
    image = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(clip), alpha=False).tobytes('png')
    path = HERE / 'originals' / f'page-{page_number}-{label}-4x.png'
    assert not path.exists() or path.read_bytes() == image
    if not path.exists(): path.write_bytes(image)
    evidence.append(dict(pdf_page=page_number, printed_page=page_number, clip=clip,
        path=str(path.relative_to(HERE)), sha256=hashlib.sha256(image).hexdigest()))
vectors = []
for page_number in (55, 56, 57):
    page = doc[page_number - 1]
    for drawing in page.get_drawings():
        rect = list(drawing['rect'])
        if rect[2] - rect[0] < 25 and rect[3] - rect[1] < 4:
            vectors.append(dict(pdf_page=page_number, rect=rect, width=drawing['width'],
                stroke_color=drawing['color'], items=repr(drawing['items'])))
(HERE / 'detail-font-evidence.json').write_text(json.dumps(dict(original_pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
    immutable_crops=evidence, short_lowline_vector_candidates=vectors,
    policy='Vector low lines are reviewed separately from embedded PUA underdots. No phonemic interpretation inferred.'), indent=2) + '\n')
