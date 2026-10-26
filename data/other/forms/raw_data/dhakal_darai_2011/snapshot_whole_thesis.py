"""Preserve every native-text page and unresolved font mark before semantic recovery.

This is an evidence snapshot, not lexical output or a completeness claim. It uses
only the retained PDF, keeps all source text/fonts/positions, and creates no renders.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import fitz
from snapshot_chapters12 import decode

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / 'tmp/pdfs/darai-survey/dhakal-2011.pdf'
EXPECTED_SHA256 = '6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert digest(PDF) == EXPECTED_SHA256, 'Unexpected original thesis'
    output = HERE / 'whole-native-pages.jsonl'
    temporary = output.with_suffix('.jsonl.tmp')
    counts = Counter()
    private = Counter()
    page_records = []
    with fitz.open(PDF) as document, temporary.open('w') as out:
        assert len(document) == 486
        for index, page in enumerate(document):
            lines = []
            image_blocks = 0
            for block in page.get_text('rawdict')['blocks']:
                if block['type'] != 0:
                    image_blocks += 1
                    continue
                for line in block['lines']:
                    spans = []
                    for span in line['spans']:
                        raw = ''.join(c['c'] for c in span['chars'])
                        marks = []
                        for offset, char in enumerate(span['chars']):
                            if 0xE000 <= ord(char['c']) <= 0xF8FF:
                                private[(span['font'], f"U+{ord(char['c']):04X}")] += 1
                                marks.append({'offset': offset, 'codepoint': f"U+{ord(char['c']):04X}",
                                              'bbox': char['bbox'], 'origin': char['origin']})
                        spans.append({'raw': raw, 'decoded': decode(raw, span['font']),
                                      'font': span['font'], 'size': span['size'], 'bbox': span['bbox'],
                                      'origin': span['origin'], 'private_glyph_positions': marks})
                    lines.append({'line': len(lines) + 1, 'bbox': line['bbox'], 'spans': spans})
            record = {'pdf_page': index + 1, 'printed_page': index - 21 if index >= 22 else None,
                      'page_bbox': list(page.rect), 'image_blocks': image_blocks, 'lines': lines}
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts.update(pages=1, native_lines=len(lines), native_spans=sum(len(x['spans']) for x in lines), image_blocks=image_blocks)
            page_records.append({'pdf_page': index + 1, 'printed_page': record['printed_page'],
                                 'native_lines': len(lines), 'image_blocks': image_blocks,
                                 'status': 'Evidence captured; lexical/control disposition requires semantic review'})
    temporary.replace(output)
    summary = {'source_sha256': EXPECTED_SHA256, 'snapshot_sha256': digest(output),
               'snapshot_bytes': output.stat().st_size, 'counts': dict(counts),
               'extraction': 'PyMuPDF rawdict; complete page text; known font-specific decoder only; unknown private glyphs explicit',
               'private_glyph_inventory': [{'font': f, 'codepoint': c, 'count': n} for (f, c), n in sorted(private.items())],
               'pages': page_records, 'canonical_modified': False,
               'status': 'All 486 pages captured; not a completed lexical inventory'}
    (HERE / 'whole-native-snapshot-census-20260927.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': dict(counts), 'bytes': output.stat().st_size, 'sha256': summary['snapshot_sha256']}))


if __name__ == '__main__':
    main()
