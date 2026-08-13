"""Print a reproducible random HTML-versus-CSV DEDR audit sample."""

import argparse
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


HERE = Path(__file__).parent
APPENDIX_PAGE = 509


def source_entries():
    with (HERE / 'dedr.pickle').open('rb') as stream:
        pages = pickle.load(stream)
    entries = {}
    for page_number, page in enumerate(pages, 1):
        for chunk in page.split('<number>')[1:]:
            soup = BeautifulSoup('<number>' + chunk, 'html.parser')
            number = soup.find('number')
            if not number:
                continue
            key = ('a' if page_number >= APPENDIX_PAGE else '') + number.get_text(strip=True)
            number.decompose()
            entries['d' + key] = soup.get_text(' ', strip=True)
    return entries


def converted_rows():
    rows = defaultdict(list)
    with (HERE / 'dedr_new.csv').open() as stream:
        for row in csv.reader(stream):
            rows[row[1]].append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260812)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    entries = source_entries()
    rows = converted_rows()
    sample = random.Random(args.seed).sample(sorted(entries), args.count)
    print('sample:', ', '.join(sample))
    for key in sample:
        print(f'\n### {key}\nHTML: {entries[key]}\nROWS:')
        for row in rows.get(key, []):
            tags = row[14] if len(row) > 14 else ''
            print(' | '.join((row[0], row[2], row[3], row[6], row[7], tags)))


if __name__ == '__main__':
    main()
