"""Prepare or install reviewed Pattapu source files; never build the database."""
import argparse
import csv
import json
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = 'rebbavarapu2013pattapu'
DIALECT = 'dialect:Pattapu:pattapu_ethamukkala:Ethamukkala'
PRONOUNS = {
    202: ('I', ['pron', '1sg']),
    203: ('you', ['pron', '2sg', 'informal']),
    204: ('you', ['pron', '2sg', 'formal']),
    205: ('he', ['pron', '3sg', 'm']),
    206: ('she', ['pron', '3sg', 'f']),
    207: ('we', ['pron', '1pl']),
    208: ('we', ['pron', 'first-person', 'du']),
    209: ('you', ['pron', '2pl']),
    210: ('they', ['pron', '3pl']),
}

def build():
    cells = [json.loads(x) for x in (ROOT/'font-recovered-scaffold.jsonl').read_text().splitlines()]
    reviews = {r['item']: r for r in map(json.loads, (ROOT/'visual-review.jsonl').read_text().splitlines())}
    rows, audit = [], []
    for cell in cells:
        item = cell['item']
        review = reviews[item]
        assert review['recovered_text'] == cell['raw_text']
        prompt, separator, response = cell['raw_text'].partition('-')
        assert separator and prompt
        response = response.strip()
        record = {'item': item, 'raw': cell, 'review': review, 'status': 'proposed', 'rows': []}
        if review['status'] != 'glyphs-compared-with-render':
            record['status'] = 'excluded-unanswered' if not response else 'withheld-transcription'
            audit.append(record)
            continue
        assert response and '\ue000' not in response
        gloss, tags = PRONOUNS.get(item, (prompt.strip(), []))
        tags = list(tags) + [DIALECT]
        if '\u0361' in response:
            tags.append('uncertain')
            record['issues'] = ['transcription:source-tie-placement-preserved-without-phonological-reinterpretation']
        if prompt.endswith('!'):
            gloss = gloss.rstrip('!')
            tags += ['verb', 'impv']
        if 151 <= item <= 164:
            tags.append('num')
        # These are complete elicited clauses, not isolated verb stems.
        if item in (183,184,185,186,187,188,191,192,193,194,200,201):
            tags.append('multiword-expression')
        readings = [x.strip() for x in response.split(',')]
        assert len(readings) == (2 if item in (11,96) else 1)
        for i, form in enumerate(readings, 1):
            key = f'pattapu-iso2013:item:{item}:reading:{i}'
            locator = f'p. {cell["printed_page"]}, col. {cell["column"]}, item {item}'
            # Comma-separated responses attest alternatives but do not assert
            # a derivational direction or a morphological variant relationship.
            row = ['Pattapu', '', unicodedata.normalize('NFC', form), gloss,
                   '', '', '', f'{SOURCE}[{locator}]', '', '', key, '', '', '', ' '.join(tags)]
            rows.append(row)
            record['rows'].append(row)
        audit.append(record)
    assert len(cells) == len(audit) == 210
    assert len(rows) == 201
    assert len({r[10] for r in rows}) == len(rows)
    return rows, audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--install', action='store_true', help='Install source CSV, settings and profile only')
    args = parser.parse_args()
    if not args.output and not args.install:
        parser.error('provide --output or --install')
    args.output = args.output or ROOT
    rows, audit = build()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'20260922-pattapu-iso.csv').open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    (args.output/'audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in audit))
    if args.install:
        for suffix in ('csv', 'yaml'):
            src = (args.output if suffix == 'csv' else ROOT)/f'20260922-pattapu-iso.{suffix}'
            shutil.copyfile(src, ROOT.parent.parent/src.name)
        shutil.copyfile(ROOT/'pattapu-iso.txt', ROOT.parents[4]/'conversion/pattapu-iso.txt')
    print(f'{len(rows)} source rows; {len(audit)} audited prompts; installed={args.install}; no database build')
