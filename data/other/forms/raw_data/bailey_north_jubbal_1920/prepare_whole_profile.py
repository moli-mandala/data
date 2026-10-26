"""Preservation-only preview profile and corpus coverage; no canonical edits."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[4]


def main():
    with (HERE / 'whole-unfrozen-preview.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
    profile = DATA / 'conversion/bailey-north-jubbal-1920.txt'
    with profile.open(newline='') as stream:
        existing = list(csv.DictReader(stream, delimiter='\t'))
    mapping = {r['Grapheme']: r['IPA'] for r in existing}
    additions = sorted({c for row in rows for c in unicodedata.normalize('NFC', row[2])} - set(mapping))
    for char in additions:
        assert not char.isspace() and char != '\ufffd'
        mapping[char] = char.lower().replace('w', 'v').replace('ṅ', 'ŋ')
    decomposed_aliases = {}
    for grapheme, value in list(mapping.items()):
        decomposed = unicodedata.normalize('NFD', grapheme)
        if decomposed != grapheme:
            assert decomposed not in mapping or mapping[decomposed] == value
            decomposed_aliases[decomposed] = value
    mapping.update(decomposed_aliases)
    target = HERE / 'whole-preview-profile.txt'
    with target.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['Grapheme', 'IPA'])
        writer.writerows(mapping.items())
    tokenizer = Tokenizer(str(target))
    for row in rows:
        original = unicodedata.normalize('NFC', row[2])
        actual = unicodedata.normalize('NFC', tokenizer(original, column='IPA').replace(' ', '').replace('#', ' '))
        expected = original.lower().replace('w', 'v').replace('ṅ', 'ŋ')
        assert actual == expected, (row[10], original, actual, expected)
    report = dict(status='preview preservation-profile coverage passed; actual parser and editorial acceptance pending',
                  rows_checked=len(rows), input_normalization='NFC literals and explicit NFD aliases; capitals map to lowercase display only',
                  decomposed_grapheme_aliases=decomposed_aliases,
                  added_symbols=[dict(symbol=c, unicode_name=unicodedata.name(c), mapping=mapping[c]) for c in additions],
                  existing_policy='w to v and ṅ to ŋ; all other source symbols preserved',
                  breath_policy='Preserve both historical U+2018 and new original-image U+2019 literally; no phonemic mapping or silent unification.',
                  profile_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  preview_sha256=hashlib.sha256((HERE/'whole-unfrozen-preview.csv').read_bytes()).hexdigest(),
                  canonical_changes=False)
    (HERE/'whole-preview-profile-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
