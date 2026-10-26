"""Check frozen evidence and render original pages for root's independent audit."""
import hashlib
import json
import argparse
from pathlib import Path
import fitz

P = Path(__file__).resolve().parent
ROOT = P.parents[6]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='.')
    args = parser.parse_args()
    base = P / args.revision
    manifest = json.loads((base/'frozen-manifest.json').read_text())
    sample = json.loads((base/'independent-sample-20260927.json').read_text())
    assert sha(base/'frozen-manifest.json') == sample['frozen_manifest_sha256']
    for name, digest in manifest['files'].items():
        assert sha(base/name) == digest, name
    pdf = ROOT/'tmp/pdfs/turi-2024/publisher-current.pdf'
    assert sha(pdf) == manifest['original_sha256']
    doc = fitz.open(pdf)
    assert len(doc) == 42
    out = ROOT/'tmp/turi-whole-root-audit'
    out.mkdir(exist_ok=True)
    pages = sorted({u['printed_page'] for u in sample['selected']})
    for page in pages:
        doc[page-261].get_pixmap(matrix=fitz.Matrix(2,2)).save(str(out/f'p{page}.png'))
    print(f'Frozen inputs verified; rendered {len(pages)} pages: {pages}')


if __name__ == '__main__':
    main()
