"""Account for single-quoted chapter spans before accepting lexical records.

This is a review inventory, not an importer. Quoted scholarly prose and sentence
translations are intentionally visible and must receive explicit dispositions.
"""
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from decode import decode_char

HERE = Path(__file__).resolve().parent


def quoted_spans(text):
    start = None
    for i, char in enumerate(text):
        if char not in "'‘’":
            continue
        before = text[i - 1] if i else ' '
        after = text[i + 1] if i + 1 < len(text) else ' '
        # Preserve apostrophes inside possessives and contractions.
        if before.isalpha() and after.isalpha():
            continue
        if start is None:
            if char in "'‘" and (before.isspace() or before in '/('):
                start = i
        elif char in "'’":
            yield start, i + 1
            start = None
    if start is not None:
        yield start, None


def build():
    records = []
    for path in sorted((HERE / 'evidence').glob('p*-glyphs.json')):
        page = int(path.name[1:4])
        chars = json.loads(path.read_text())
        pieces, offsets = [], []
        for index, char in enumerate(chars):
            piece = decode_char(char)
            pieces.append(piece)
            offsets.extend([index] * len(piece))
        text = ''.join(pieces)
        for start, end in quoted_spans(text):
            first = offsets[start]
            before = text[max(0, start - 200):start]
            match = re.search(r'/([^/]{1,100})/\s*$', before)
            form = match.group(1).strip() if match else ''
            kind = 'slash-delimited-form' if match else 'needs-context-review'
            if end is None:
                kind = 'unclosed-quotation'
            last = offsets[end - 1] if end is not None else len(chars) - 1
            glyphs = chars[first:last + 1]
            records.append({
                'key': f'dhakal2011:p{page}:g{first}', 'printed_page': page,
                'pdf_page': page + 22, 'glyph_start': first, 'glyph_end': last,
                'bbox': [min(c['x0'] for c in glyphs), min(c['top'] for c in glyphs),
                         max(c['x1'] for c in glyphs), max(c['bottom'] for c in glyphs)],
                'raw_quote': text[start:end],
                'gloss_candidate': unicodedata.normalize('NFC', text[start + 1:end - 1]) if end else '',
                'form_candidate': unicodedata.normalize('NFC', form),
                'preceding_context': unicodedata.normalize('NFC', before),
                'following_context': unicodedata.normalize('NFC', text[end:end + 140]) if end else '',
                'kind': kind, 'status': 'unreviewed',
            })
    return records


if __name__ == '__main__':
    rows = build()
    (HERE / 'inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'quoted_spans': len(rows), 'kinds': dict(Counter(r['kind'] for r in rows)),
                      'installed_rows': 0}))
