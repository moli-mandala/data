"""Extract the complete CUJ Asur draft dictionary by physical entry boundaries.

No installation: output is a source snapshot for a subsequent semantic parser.
Headword anchors are 9pt Kohinoor Devanagari Bold at the two paragraph margins.
Continuation text carries across columns/pages until the next anchored headword.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE_FONT = 'KohinoorDevanagariBold'


def native_text(chars):
    """Undo the PDF's pre-base-i ToUnicode syllable duplication, not spelling.

    A 291/1000-em pre-base vowel glyph is wrongly mapped to a whole syllable
    (e.g. रि), followed by the actual base consonant glyph. Decode the narrow
    glyph as ि and move it after the following Devanagari consonant cluster.
    Leave unknown glyphs and every other source spelling untouched for review.
    """
    pieces = []
    for char in sorted(chars, key=lambda c: c.get('source_order', 0)):
        text = char['text']
        if text.endswith('ि') and len(text) > 1 and abs(char['width'] / char['size'] - .291) < .005:
            text = '\ue001'
        # Embedded Type1 /Encoding names CID52 /uni0925 (थ), while its
        # context-contaminated ToUnicode value is र्थ. CID53 /bad702 is the
        # separately drawn repha, verified against all eight affected heads.
        if text == 'र्थ' and abs(char['width'] / char['size'] - .707) < .005:
            text = 'थ'
        if text == '(cid:53)':
            text = '\ue000'
        if text == 'ो' and abs(char['width'] / char['size'] - .660) < .005:
            text = '◌'  # CID117 is /uni25CC in the embedded Type1 Encoding.
        pieces.append(text)
    value = ''.join(pieces)
    value = re.sub(r'\ue001([क-हक़-य़](?:़)?(?:्[क-हक़-य़](?:़)?)*)', r'\1ि', value)
    return re.sub(r'([क-हक़-य़](?:़)?(?:्[क-हक़-य़](?:़)?)*[ािीुूृेैोौंँः]*)\ue000', r'र्\1', value)


def ordered_words(page):
    """Read two-column bands separated by full-width alphabet headings."""
    headings = sorted({c['top'] for c in page.chars
                       if NATIVE_FONT in c['fontname'] and round(c['size']) == 14
                       and 170 < c['x0'] < 200})
    boundaries = [65] + [y - 2 for y in headings if 65 < y < 565] + [565]
    for top, bottom in zip(boundaries, boundaries[1:]):
        for column, (left, right) in enumerate([(45.0, 190), (204.0, 364)], 1):
            for word in page.within_bbox((left, top, right, bottom)).extract_words(
                    x_tolerance=1, y_tolerance=3, return_chars=True,
                    extra_attrs=['fontname', 'size']):
                yield column, left, word


def extract(pdf):
    import pdfplumber

    manifest = json.loads((HERE / 'manifest.json').read_text())
    if hashlib.sha256(Path(pdf).read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Input PDF differs from pinned university draft')
    record = None
    with pdfplumber.open(pdf) as doc:
        if len(doc.pages) != 110:
            raise ValueError('Expected 110 pages')
        for page_index in range(5, 110):
            page = doc.pages[page_index]
            # Keep PDF draw order: geometric x-sort incorrectly moves the
            # anusvara in अंगुर onto ग, yielding अगंुर.
            for order, char in enumerate(page.chars):
                char['source_order'] = order
            for column, left, word in ordered_words(page):
                    native = NATIVE_FONT in word['fontname']
                    anchor = (native and round(word['size']) == 9
                              and abs(word['x0'] - left - .1) < .15)
                    if anchor:
                        if record:
                            yield record
                        y = round(word['top'], 1)
                        record = {
                            'entry_key': f'cujasur2020:p{page_index-4}:c{column}:y{y:.1f}',
                            'printed_page': page_index - 4, 'pdf_page': page_index + 1,
                            'column': column, 'top': y, 'tokens': [],
                        }
                    # Alphabet headings are 14pt; running headers are outside bbox.
                    if not record or word['size'] > 10:
                        continue
                    token = {
                        'raw': word['text'], 'text': native_text(word['chars']) if native else word['text'],
                        'font': word['fontname'].split('+')[-1], 'size': round(word['size'], 2),
                        'pdf_page': page_index + 1, 'column': column,
                        'x0': round(word['x0'], 3), 'x1': round(word['x1'], 3),
                        'top': round(word['top'], 3), 'bottom': round(word['bottom'], 3),
                    }
                    if native:
                        token['glyphs'] = [
                            {'text': c['text'], 'x': round(c['x0'], 3),
                             'width': round(c['width'], 3), 'size': round(c['size'], 3),
                             'source_order': c['source_order']}
                            for c in word['chars']]
                    record['tokens'].append(token)
            page.close()
        if record:
            yield record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = 0
    with args.output.open('w') as out:
        for record in extract(args.pdf):
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    print(f'{count} physical source entries extracted; semantic parsing still required')
