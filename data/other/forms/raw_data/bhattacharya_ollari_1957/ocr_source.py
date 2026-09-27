"""Reproduce an untrusted OCR scaffold for the complete Ollari vocabulary.

One page and one OCR thread at a time. This never emits installed forms.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def extract(pdf, output):
    import pdfplumber
    manifest = json.loads((HERE / 'manifest.json').read_text())
    with pdf.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('PDF differs from pinned library scan')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_THREAD_LIMIT='1')
    ledger = []
    with pdfplumber.open(pdf) as document, tempfile.TemporaryDirectory() as temp:
        if len(document.pages) != 93:
            raise ValueError('Expected 93 PDF pages')
        for pdf_page in range(59, 89):
            page = document.pages[pdf_page - 1]
            if page.chars:
                raise ValueError('Unexpected native text: reassess extraction path')
            png = Path(temp) / 'page.png'
            page.to_image(resolution=300).save(png)
            stem = output / f'p{pdf_page - 11:02d}'
            subprocess.run(['tesseract', str(png), str(stem), '-l', 'eng',
                            '--psm', '3', 'txt', 'tsv'], env=env, check=True,
                           stdout=subprocess.DEVNULL)
            ledger.append({'pdf_page': pdf_page, 'printed_page': pdf_page - 11,
                           'width_points': page.width, 'height_points': page.height,
                           'text_sha256': hashlib.sha256(stem.with_suffix('.txt').read_bytes()).hexdigest(),
                           'tsv_sha256': hashlib.sha256(stem.with_suffix('.tsv').read_bytes()).hexdigest(),
                           'status': 'unreviewed-OCR-not-installable'})
            page.close()
            print(f'printed p.{pdf_page - 11}: OCR scaffold saved', flush=True)
    (output / 'pages.json').write_text(json.dumps(ledger, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'ocr')
    args = parser.parse_args()
    extract(args.pdf, args.output)
