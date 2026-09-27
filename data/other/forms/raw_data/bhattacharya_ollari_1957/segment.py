"""Candidate physical entry boundaries, never an installed lexical parser."""
import collections
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def page_lines(page):
    groups = collections.defaultdict(list)
    with (HERE / 'ocr' / f'p{page:02d}.tsv').open() as stream:
        source = list(csv.DictReader(stream, delimiter='\t'))
    width = int(source[0]['width'])
    # Printed p.52 is skewed: lower right headwords cross the geometric midpoint.
    # Visually verified gutter at x=1300 in the pinned 300-dpi OCR coordinates.
    column_cut = 1300 if page == 52 else width / 2
    for word in source:
        if word['level'] == '5' and word['text'].strip():
            groups[tuple(word[k] for k in ['block_num', 'par_num', 'line_num'])].append(word)
    lines = []
    for words in groups.values():
        x, y = min(int(w['left']) for w in words), min(int(w['top']) for w in words)
        text = ' '.join(w['text'] for w in words)
        if 'COMPARATIVE VOCABULARY' in text or text.strip('0123456789 :') == 'OLLARI' or not re.search('[A-Za-z]', text):
            continue
        lines.append({'text': text, 'left': x, 'top': y,
                      'right': max(int(w['left']) + int(w['width']) for w in words),
                      'bottom': max(int(w['top']) + int(w['height']) for w in words),
                      'column': 1 if x < column_cut else 2,
                      'printed_page': page, 'words': words})
    manual = json.loads((HERE / 'manual-lines.json').read_text())
    lines.extend(r for r in manual if r['printed_page'] == page)
    return sorted(lines, key=lambda r: (r['column'], r['top']))


def capital_head(line):
    first = line['text'].split()[0]
    letters = [c for c in first if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= .8 and first[0] not in '[('


def lexical_start(line):
    return capital_head(line) or bool(re.match(
        r'[^\[\];]{1,60},\s*(?:[sS5][b6]\.|[vV¥][b5d]\.|[aA@]d[jyv]+\.|pl\.?|pron\.|num\.|conj\.|infl|\(obl\.)',
        line['text']))


def extract():
    records, current = [], None
    overrides = json.loads((HERE / 'boundary-overrides.json').read_text())
    forced = {(r['printed_page'], r['column'], r['top']) for r in overrides}
    for page in range(48, 78):
        lines = page_lines(page)
        for column in [1, 2]:
            part = [r for r in lines if r['column'] == column]
            candidates = [r for r in part if lexical_start(r) and (page != 48 or r['top'] > 1250)]
            third = max(1, len(candidates) // 3)
            first = min(candidates[:third], key=lambda r: r['left'])
            last = min(candidates[-third:], key=lambda r: r['left'])
            slope = (last['left'] - first['left']) / max(1, last['top'] - first['top'])
            for line in part:
                margin = first['left'] + slope * (line['top'] - first['top'])
                anchor = lexical_start(line) and abs(line['left'] - margin) < 42 and (page != 48 or line['top'] > 1250)
                anchor = anchor or (page, column, line['top']) in forced
                anchor = anchor or line.get('origin') == 'manual-scan-transcription'
                if anchor:
                    if current:
                        records.append(current)
                    current = {'entry_key': line.get('entry_key', f'ollari1957:p{page}:c{column}:y{line["top"]}'),
                               'printed_page': page, 'pdf_page': page + 11,
                               'column': column, 'top': line['top'], 'lines': [],
                               'status': 'candidate-boundary-unreviewed'}
                if current:
                    current['lines'].append(line)
    if current:
        records.append(current)
    return records


if __name__ == '__main__':
    records = extract()
    with (HERE / 'candidate-entries.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('Candidate boundaries, NOT verified source-entry count:', len(records))
    for page in range(48, 78):
        entries = [r for r in records if r['printed_page'] == page]
        print(page, len(entries), ' | '.join(r['lines'][0]['text'].split(',')[0] for r in entries))
