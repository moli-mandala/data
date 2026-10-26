"""Whole chapter/page/native-font census; no accepted lexical transcription."""
import hashlib
import json
import re
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/SOURCE_INGESTION_CHECKLIST.md').exists())
PDF = ROOT / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(PDF.read_bytes()) == '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
doc = fitz.open(PDF)
originals = HERE / 'originals'
originals.mkdir(exist_ok=True)
pages = []
for printed in range(55, 59):
    page = doc[printed - 1]
    text = page.get_text()
    (HERE / f'page-{printed}-native.txt').write_text(text)
    raw = page.get_text('rawdict')
    (HERE / f'page-{printed}-rawdict.json').write_text(json.dumps(raw, ensure_ascii=False))
    image = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False).tobytes('png')
    path = originals / f'page-{printed}-150dpi.png'
    assert not path.exists() or sha(path.read_bytes()) == sha(image), 'Never overwrite a changed original render'
    if not path.exists():
        path.write_bytes(image)
    private = sorted({c for c in text if 0xE000 <= ord(c) <= 0xF8FF})
    lines = []
    for block in raw['blocks']:
        for line in block.get('lines', []):
            lines.append({'rect': line['bbox'], 'text': ''.join(c['c'] for span in line['spans'] for c in span['chars']),
                'fonts': sorted({span['font'] for span in line['spans']})})
    pages.append(dict(printed_page=printed, pdf_page=printed, native_text_sha256=sha(text.encode()),
        original_image=str(path.relative_to(HERE)), original_image_sha256=sha(image),
        PUA=[{'character': c, 'codepoint': f'U+{ord(c):04X}'} for c in private], native_lines=lines,
        acceptance='unaccepted native extraction; original glyph review required'))
mentions = []
for number, page in enumerate(doc, 1):
    text = page.get_text()
    for match in re.finditer(r'mullu\s*k[ua]rumba|mullukurumba|mullu\s*kuruman', text, re.I):
        mentions.append({'pdf_page': number, 'context': text[max(0, match.start() - 180):match.end() + 300],
            'inside_chapter': 55 <= number <= 58})
(HERE / 'page-native-font-census.json').write_text(json.dumps(dict(status='scope census, not accepted extraction',
    original_pdf=str(PDF.relative_to(ROOT)), source_pdf_sha256=sha(PDF.read_bytes()),
    physical_pdf_pages=len(doc), chapter_printed_pages=[55, 56, 57, 58], page_mapping='printed folio equals one-based PDF page; visual verification required',
    pages=pages, whole_book_name_mentions=mentions, canonical_changes=False), ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'chapter_pages': 4, 'whole_book_mentions': len(mentions), 'canonical_changes': False}))


if __name__ == '__main__':
    pass
