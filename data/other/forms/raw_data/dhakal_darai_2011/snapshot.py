"""Freeze native glyph evidence for the complete Darai phonology chapter.

No lexical rows are installed by this acquisition command. Glyph order remains
the PDF's original order; overlapping combining marks require geometric review.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PDF = ROOT.parent / 'tmp/pdfs/darai-survey/dhakal-2011.pdf'
SHA256 = '6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf'


def snapshot(pdf, output):
    if not pdf.is_file():
        raise FileNotFoundError(f'Acquire the university dissertation first: {pdf}')
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Dissertation differs from pinned university PDF')
    output.mkdir(parents=True, exist_ok=True)
    inventory = Counter()
    manifest = {'pdf_sha256': SHA256, 'pdf_pages': 486, 'printed_pages': [42, 76],
                'pdf_range': [64, 98], 'status': 'native glyph snapshot; not parsed or installed',
                'pages': []}
    with pdfplumber.open(pdf) as doc:
        assert len(doc.pages) == 486
        assert 'CHAPTER 3' in doc.pages[63].extract_text()
        assert 'CHAPTER 4' in doc.pages[98].extract_text()
        for index in range(63, 98):
            page = doc.pages[index]
            printed = index - 21
            chars = [{key: c[key] for key in ('text', 'fontname', 'x0', 'x1', 'top', 'bottom', 'size')}
                     for c in page.chars]
            for c in chars:
                if any(0xe000 <= ord(ch) <= 0xf8ff for ch in c['text']):
                    inventory[c['fontname'], c['text']] += 1
            glyph_path = output / f'p{printed:03}-glyphs.json'
            text_path = output / f'p{printed:03}-text.txt'
            glyph_path.write_text(json.dumps(chars, ensure_ascii=False, separators=(',', ':')) + '\n')
            text_path.write_text((page.extract_text() or '') + '\n')
            manifest['pages'].append({'printed_page': printed, 'pdf_page': index + 1,
                'width': page.width, 'height': page.height, 'glyph_count': len(chars),
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [glyph_path, text_path]}})
    manifest['private_use_glyphs'] = [{'font': font, 'glyph': glyph, 'count': count}
                                    for (font, glyph), count in sorted(inventory.items())]
    (output / 'snapshot.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=PDF)
    parser.add_argument('--output', type=Path, default=HERE / 'evidence')
    args = parser.parse_args()
    result = snapshot(args.pdf, args.output)
    print(f"{len(result['pages'])} pages; {sum(p['glyph_count'] for p in result['pages'])} native glyphs; no forms installed")
