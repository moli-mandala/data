"""Extract only the numbered Koda column of the official West Bengal LSI table.

The PDF remains in the local tmp cache. The 500-cell positional inventory is
written separately from reviewed, potentially installable lexical readings.
"""
import hashlib
import json
import re
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / 'tmp/pdfs/census-koda/LSI_WB_PART_I.pdf'
SHA256 = 'd3d1f935612bad52cb79699c07702aa4efafd26a872163d8998782fde4639a63'
OUT = HERE / 'cells.jsonl'


def extract():
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Official LSI West Bengal PDF SHA-256 changed')
    cells = []
    with pdfplumber.open(PDF) as doc:
        for page_number in range(751, 777):
            page = doc.pages[page_number - 1]
            if page.width < 800 or page.height < 590:
                raise ValueError(f'Unexpected table geometry on PDF p{page_number}')
            words = page.extract_words(x_tolerance=1, y_tolerance=3)
            if not any(w['text'] == 'KODA/KORA' for w in words):
                raise ValueError(f'Koda table header missing from PDF p{page_number}')
            numbers = [w for w in words if 145 < w['x0'] < 190
                       and 120 < w['top'] < 550 and re.fullmatch(r'\d+', w['text'])]
            for index, number in enumerate(numbers):
                top = number['top']
                bottom = numbers[index + 1]['top'] if index + 1 < len(numbers) else 550

                def column(x0, x1):
                    selected = [w for w in words if x0 <= w['x0'] < x1
                                and top - 3 <= w['top'] < bottom - 3]
                    return ' '.join(w['text'] for w in
                                    sorted(selected, key=lambda w: (round(w['top']/3)*3, w['x0'])))

                cells.append({'prompt': int(number['text']), 'pdf_page': page_number,
                              'printed_page': page_number - 20,
                              'raw_english': column(195, 325),
                              'raw_koda': column(470, 620),
                              'top': round(top, 2), 'bottom': round(bottom, 2),
                              'source_status': 'unreviewed positional extraction'})
    if [c['prompt'] for c in cells] != list(range(1, 501)):
        raise ValueError('Koda table no longer accounts for exactly prompts 1–500')
    return cells


if __name__ == '__main__':
    cells = extract()
    OUT.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cells))
    print(f'{len(cells)} numbered Koda cells extracted from 26 printed table pages')
