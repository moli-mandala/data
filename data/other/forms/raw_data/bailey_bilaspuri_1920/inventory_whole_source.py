"""Freeze the full section's embedded OCR as locating evidence, never accepted text."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz

SOURCE_SHA256 = '7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'whole-source-text-inventory.jsonl')
    args = parser.parse_args()
    digest = hashlib.file_digest(args.pdf.open('rb'), 'sha256').hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'Unexpected source PDF: {digest}')
    with fitz.open(args.pdf) as document, args.output.open('w') as out:
        for printed in range(231, 249):
            out.write(json.dumps({
                'printed_page': printed,
                'pdf_page': printed + 26,
                'source_sha256': digest,
                'evidence_type': 'unreviewed embedded OCR; locator only',
                'text': document[printed + 25].get_text(),
            }, ensure_ascii=False) + '\n')
    print('18 pages inventoried; no canonical forms or database written.')


if __name__ == '__main__':
    main()
