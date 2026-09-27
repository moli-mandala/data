"""Prepare the whole Samuells article's attested Juang units; never build a database."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[4]
STEM = '20260925-samuells-juang'
SOURCE = 'samuells1856juang'
DIALECT = 'dialect:ju:samuells-1856-juang:Juang%20%28Samuells%201856%29'
OUTPUT = ROOT.parents[1] / f'{STEM}.csv'
PROFILE = DATA / 'conversion/samuells-juang-1856.txt'


def build():
    units = [json.loads(line) for line in (ROOT / 'full-reviewed-inventory.jsonl').read_text().splitlines()]
    assert len(units) == 33
    rows, audit = [], []
    for u in units:
        u = dict(u)
        forms = u['forms']
        base = u['source_unit_key']
        keys = [f'{base}:v{i}' if len(forms) > 1 else base for i in range(1, len(forms) + 1)]
        u['entry_keys'] = keys
        u['disposition'] = 'Whole attested response retained; no inferred component lemmata.'
        for i, form in enumerate(forms):
            tags = list(u['tags'])
            if ' ' in form:
                tags.append('multiword-expression')
            notes = u['notes']
            if len(forms) > 1:
                notes = ' '.join(filter(None, [notes, f'Printed alternative {i + 1}/{len(forms)} for “{u["printed_prompt"]}”.']))
                if i:
                    tags.append('alternate')
            locator = f'vocabulary item {u["item"]}' if u['section'] == 'vocabulary' else ('self-name paragraph' if u['printed_page'] == 296 else 'title paragraph')
            rows.append(['ju', '', form, u['gloss'], '', '', notes,
                         f'{SOURCE}[p. {u["printed_page"]}, {locator}]', '', '', keys[i],
                         keys[0] if i else '', '', '', ' '.join(dict.fromkeys([*tags, DIALECT]))])
        audit.append(u)
    # Keep the legacy append positions as well as the source keys. New material follows.
    legacy = list(csv.reader((ROOT / 'legacy-before-full-recovery' / f'{STEM}.csv').open()))
    old_order = {row[10]: i for i, row in enumerate(legacy)}
    rows.sort(key=lambda row: old_order.get(row[10], len(legacy)))
    assert len(rows) == len({row[10] for row in rows}) == 35
    assert [r[10] for r in rows[:21]] == [r[10] for r in legacy]
    assert all(not r[8] and not r[9] and not r[12] and not r[13] for r in rows)
    assert all(not r[11] or r[11] in {x[10] for x in rows} for r in rows)
    return rows, audit


def encoded(rows, audit):
    out = io.StringIO(newline='')
    csv.writer(out).writerows(rows)
    chars = sorted({ch for row in rows for ch in row[2]})
    # The source supplies no sound key: preserve acute accents/doubled vowels literally.
    profile = 'Grapheme\tIPA\n' + ''.join(f"{ch}\t{'#' if ch == ' ' else ch.lower().replace('w', 'v')}\n" for ch in chars)
    return {'proposal.csv': out.getvalue(),
            'proposal-audit.jsonl': ''.join(json.dumps(u, ensure_ascii=False, sort_keys=True) + '\n' for u in audit),
            'proposal-profile.txt': profile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows, audit = build()
    artifacts = encoded(rows, audit)
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in artifacts.items()}
    if args.install:
        review = json.loads((ROOT / 'independent-full-output-review-20260926.json').read_text())
        assert review['status'] in ('passed', 'pass') and review['material_errors'] == 0
        assert review['sample_size'] >= 20
        assert review['hashes'] == hashes, 'Independent review must pin the exact proposed outputs.'
        OUTPUT.write_bytes(artifacts['proposal.csv'].encode())
        (ROOT / f'{STEM}.csv').write_bytes(artifacts['proposal.csv'].encode())
        (ROOT / 'audit.jsonl').write_bytes(artifacts['proposal-audit.jsonl'].encode())
        PROFILE.write_bytes(artifacts['proposal-profile.txt'].encode())
    for name, content in artifacts.items():
        (ROOT / name).write_bytes(content.encode())
    print(json.dumps({'rows': len(rows), 'audit_units': len(audit), 'installed': args.install, 'hashes': hashes}, indent=2))


if __name__ == '__main__':
    main()
