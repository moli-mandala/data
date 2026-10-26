"""Small original-page crops for recovering drawn underlines and legacy font marks."""
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[6] / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
doc = fitz.open(PDF)
clips = {
    'pdf34-initial-table': (34, (75, 225, 460, 382)),
    'pdf35-first-tables': (35, (72, 60, 465, 300)),
    'pdf36-pronouns': (36, (75, 50, 455, 180)),
}
for name, (number, coords) in clips.items():
    doc[number - 1].get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(coords),
                                alpha=False).save(HERE / f'{name}-crop.png')
print(len(clips))
