"""Pin the census scan and preserve its OCR layer before lexical interpretation.

Run from data/: .venv/bin/python data/other/forms/raw_data/das_yerava_1987/snapshot.py
The PDF is deliberately kept outside the checked-in source package.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SHA256 = '425ed1e1d3a8de28086b066eb466746d58cb34ca55cbd6e545ab04a3a1eea456'
DEFAULT_PDF = ROOT.parent / 'tmp/pdfs/yerava-census-1981/source.pdf'
# Bibliographic, classification and lexical evidence. Values are printed pages.
PAGES = {1: 'title', 3: 'iii', 35: '3', 36: '4', 37: '5',
         95: '63', 96: '64', 97: '65', 98: '66', 175: '143', 190: '158', 191: '159', 192: '160'}


def snapshot(pdf, output):
    if not pdf.is_file():
        raise FileNotFoundError(f'Acquire the census PDF first: {pdf}')
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f'Unexpected census scan SHA256: {actual}')
    output.mkdir(parents=True, exist_ok=True)
    result = {'pdf_sha256': actual, 'pages': []}
    with pdfplumber.open(pdf) as document:
        assert len(document.pages) == 193
        result['pdf_pages'] = len(document.pages)
        result['pdf_metadata'] = document.metadata
        for number, printed in PAGES.items():
            page = document.pages[number - 1]
            text = page.extract_text() or ''
            path = output / f'pdf-{number:03}.txt'
            path.write_text(text + '\n', encoding='utf-8')
            result['pages'].append({'pdf_page': number, 'printed_page': printed,
                                    'path': path.name, 'characters': len(text),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / 'snapshot.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=DEFAULT_PDF)
    parser.add_argument('--output', type=Path, default=HERE / 'evidence')
    args = parser.parse_args()
    result = snapshot(args.pdf, args.output)
    print(f"Preserved {len(result['pages'])} evidence pages from {result['pdf_pages']}-page scan")
