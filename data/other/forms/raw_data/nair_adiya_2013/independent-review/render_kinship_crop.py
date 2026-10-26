from pathlib import Path
import fitz

root = next(p for p in Path(__file__).resolve().parents if (p / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf').exists())
doc = fitz.open(root / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf')
page = doc[6]
page.get_pixmap(matrix=fitz.Matrix(6, 6), clip=fitz.Rect(90, 636, 505, 678), alpha=False).save(Path(__file__).with_name('p7-kinship-highres.png'))
print(repr(page.get_text()[page.get_text().find('Husband'):page.get_text().find('Bibliography')]))
