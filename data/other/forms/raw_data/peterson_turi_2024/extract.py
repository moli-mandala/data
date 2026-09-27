"""Extract the pinned publisher PDF; no OCR, installation, or normalization.

Run from any directory with the PDF path. PDF glyph origins, rather than glyph
bounding boxes, assign floating accents/subscripts to the correct printed row.
The checked-in records retain exact text-layer values for subsequent review.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent


def extract(pdf):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if hashlib.sha256(Path(pdf).read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('PDF differs from the pinned canonical publisher edition')
    records = []
    with pdfplumber.open(pdf) as doc:
        if len(doc.pages) != 42:
            raise ValueError('Expected 42 PDF pages')
        for idx in range(30, 37):
            page = doc.pages[idx]
            labels = [w for w in page.extract_words(x_tolerance=1, y_tolerance=3)
                      if abs(w['x0'] - 87.874) < .1
                      and re.fullmatch(r'\d+(?:\.\d+)?[.,]?', w['text'])]
            for label in labels:
                # Match the label's first full-size glyph to recover its baseline.
                anchor = next(c for c in page.chars
                              if abs(c['x0'] - label['x0']) < .01
                              and abs(c['top'] - label['top']) < .01)
                baseline = anchor['matrix'][5]
                chars = [c for c in page.chars
                         if 109 <= c['x0'] < 370
                         and abs(c['matrix'][5] - baseline) < 5]
                def column(lo, hi):
                    return ''.join(c['text'] for c in chars
                                   if lo <= c['x0'] < hi).strip()
                records.append({
                    'printed_item': label['text'], 'pdf_page': idx + 1,
                    'printed_page': idx + 261,
                    'entry_key': f"peterson2024turi:p{idx+261}:item{label['text'].rstrip('.,')}",
                    'baseline': baseline,
                    'gloss': column(109, 180), 'raw_form': column(180, 370),
                    'glyphs': [{'text': c['text'], 'x': c['matrix'][4],
                                'baseline': c['matrix'][5], 'font': c['fontname']}
                               for c in chars],
                })
    expected = [str(i) for i in range(1, 275)]
    expected.insert(52, '52.1')
    expected[254] = '234'  # Printed typo: 'why', distinct from p.296 'they (f.)'.
    assert [r['printed_item'].rstrip('.,') for r in records] == expected
    assert len({r['entry_key'] for r in records}) == 275
    assert all(r['gloss'] and r['raw_form'] for r in records)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = extract(args.pdf)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(records)} source records; '
          f'{sum(r["raw_form"] == "-" for r in records)} unelicited cells')
