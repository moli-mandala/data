"""Pin complete chapter text evidence; renders are review aids, never accepted OCR."""
import hashlib
import json
import argparse
from pathlib import Path
import fitz

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[5]
PDF = ROOT / 'tmp/pdfs/bailey1920/bailey1920.pdf'
EXPECTED = '7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-start', type=int, default=87)
    parser.add_argument('--render-end', type=int, default=91)
    parser.add_argument('--scale', type=float, default=2)
    args = parser.parse_args()
    assert 87 <= args.render_start <= args.render_end <= 109
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
    doc = fitz.open(PDF)
    assert len(doc) == 310
    out = PACKAGE / 'whole-source-recovery'
    out.mkdir(exist_ok=True)
    pages = []
    for printed in range(87, 110):
        page = doc[printed + 25]
        pages.append(dict(printed_page=printed, pdf_page=printed+26,
                          text_inventory_only=page.get_text(),
                          blocks=page.get_text('blocks'), status='original review pending'))
    (out / 'chapter-native-inventory.jsonl').write_text(''.join(
        json.dumps(p, ensure_ascii=False)+'\n' for p in pages))
    manifest = dict(source='bailey1920kagani', pdf_sha256=EXPECTED, pdf_pages=310,
        scope='Complete Kagani chapter printed87–109, including pronunciation, grammar, interlinear parable, notes, and glossary; attributed cross-references to be inventoried.',
        pages=len(pages), existing_glossary_rows=329,
        status='whole-source recovery active; text layer is an inventory aid only',
        exclusions_policy='Account for controls and abstract grammatical notation individually; recover all source-attested forms and whole translated expressions.',
        deferred=['source review and extraction', 'independent fresh output audit',
                  'source-stage installation', 'full CLDF/build/compiled checks', 'user-triggered database/browser checks'])
    (out / 'scope-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    render = ROOT / 'tmp/kagani-whole'
    render.mkdir(exist_ok=True)
    for printed in range(args.render_start, args.render_end+1):
        doc[printed+25].get_pixmap(matrix=fitz.Matrix(args.scale,args.scale)).save(str(render / f'p{printed}.png'))
    print(f'Inventoried {len(pages)} chapter pages; rendered {args.render_start}–{args.render_end} for original review.')


if __name__ == '__main__':
    main()
