"""Pin native PDF typography beside the existing OCR; never replace its spelling.

Run with data/.venv/bin/python. Extraction is sequential and closes each page.
The scan itself is not redistributed. Native text has unreliable diacritics;
only its position and font provide structural evidence.
"""
import gzip
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('berger_layout_base', HERE.parent / 'berger_cleanup.py')
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def extract(destination):
    import pdfplumber
    assert b.sha256_path(b.legacy.DEFAULT_PDF) == b.PDF_SHA256
    with pdfplumber.open(b.legacy.DEFAULT_PDF) as doc, gzip.open(destination, 'wt', encoding='utf-8') as out:
        for raw in b.load_pages(b.CACHE_DIR):
            number = raw['pdf_page']
            if number in b.EXCLUDED_PDF_PAGES:
                continue
            page = doc.pages[number - 1]
            scale = raw['width'] / page.width
            centers = b.repair.column_starts(raw)
            words = page.extract_words(extra_attrs=['fontname'], x_tolerance=1, y_tolerance=2)
            for line in raw['lines']:
                col = b.repair.column(line['left'], centers)
                if col not in b.allowed_columns(number):
                    continue
                found = [w for w in words if abs(w['top'] * scale - line['top']) < 15
                         and b.repair.column(w['x0'] * scale, centers) == col
                         and w['x0'] * scale >= line['left'] - 20]
                found.sort(key=lambda w: w['x0'])
                out.write(json.dumps({'pdf': number, 'left': line['left'], 'top': line['top'],
                                      'ocr': line['text'], 'words': [[w['text'], w['fontname']] for w in found]},
                                     ensure_ascii=False) + '\n')
            page.close()
            if number % 40 == 0:
                print(number, flush=True)


if __name__ == '__main__':
    extract(HERE / 'layout.jsonl.gz')
