"""Reproduce the complete reviewed Yerava table; default output is a preview."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
STEM = '20260921-das-yerava'
SOURCE = 'das1987yerava'
LECTS = {
    'Panjiri Yerava': ('Ravula', 'ravula_panjiri_kodagu'),
    'Pani Yerava': ('Paniya', 'paniya_pani_kodagu'),
}


def build():
    with (HERE / 'reviewed.tsv').open(encoding='utf-8') as stream:
        table = list(csv.DictReader(stream, delimiter='\t'))
    assert [(r['Page'], int(r['Item'])) for r in table] == (
        [('65', i) for i in range(1, 28)] + [('66', i) for i in range(1, 16)])
    rows, audit = [], []
    for record in table:
        for lect, (language, dialect) in LECTS.items():
            cell_key = f"{SOURCE}:p{record['Page']}:i{record['Item']}:{dialect}"
            forms = record[lect].split(', ')
            keys = []
            for index, form in enumerate(forms, 1):
                key = f'{cell_key}:response:{index}'
                keys.append(key)
                # Printed popular spelling is not a phonemic analysis. No
                # unstated vowel length, retroflexion or aspiration is inferred.
                notes = ('The printed brace groups all three English kinship labels.'
                         if record['Page'] == '65' and record['Item'] == '3' else '')
                rows.append([language, '', form, record['English'], '', '', notes,
                             f"{SOURCE}[p. {record['Page']}, table item {record['Item']}, {lect}]",
                             '', '', key, '', '', '',
                             f"dialect:{language}:{dialect}:{lect.replace(' ', '-')}"])
            audit.append({'cell_key': cell_key, 'printed_page': int(record['Page']),
                          'pdf_page': int(record['Page']) + 32,
                          'item': int(record['Item']), 'source_lect': lect,
                          'language': language, 'dialect': dialect,
                          'source_cell': record[lect], 'source_gloss': record['English'],
                          'review': record['Review'], 'entry_keys': keys,
                          'forms': forms, 'status': 'reviewed',
                          'relations': 'none: co-equivalents do not assert variants or ancestry'})
    assert len(rows) == 90 and len(audit) == 84
    assert len({r[10] for r in rows}) == 90
    return rows, audit


def generate(output):
    rows, audit = build()
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'{STEM}.csv').open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    sample = random.Random(2026092106).sample(audit, 20)
    acceptance = {'seed': 2026092106, 'unit': 'language cell',
                  'reviewed_sha256': hashlib.sha256((HERE / 'reviewed.tsv').read_bytes()).hexdigest(),
                  'output_sha256': hashlib.sha256((output / f'{STEM}.csv').read_bytes()).hexdigest(),
                  'sample': sample}
    (output / 'acceptance-sample.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + '\n')
    return acceptance


def install():
    from pybtex.database import parse_file
    result = json.loads((HERE / 'acceptance-results.json').read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        sample = generate(output)
        if result['material_errors'] != 0 or result['sample'] != sample:
            raise ValueError('Acceptance audit does not cover current source/output')
        bib_path = ROOT / 'cldf/sources.bib'
        existing = parse_file(str(bib_path)).entries
        entry = parse_file(str(HERE / 'source.bib')).entries[SOURCE]
        if SOURCE in existing and existing[SOURCE] != entry:
            raise ValueError('Conflicting bibliography entry')
        dialect_path = ROOT / 'cldf/dialects.csv'
        with dialect_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            dialects = {r['ID']: r for r in reader}
        additions = []
        for lect, (language, dialect) in LECTS.items():
            row = dict.fromkeys(fields, '')
            row.update(ID=dialect, Tag=f"dialect:{language}:{dialect}:{lect.replace(' ', '-')}",
                       Language_ID=language, Source_Language_ID=lect, Name=lect,
                       Clade='S. Dravidian I', Location='Kodagu district, Karnataka; table gives no specific village')
            if dialect in dialects:
                if dialects[dialect] != row:
                    raise ValueError(f'Conflicting dialect: {dialect}')
            else:
                additions.append(row)
        # Validate shared-record collisions before changing any canonical files.
        if SOURCE not in existing:
            with bib_path.open('a') as stream:
                stream.write('\n' + (HERE / 'source.bib').read_text())
        if additions:
            with dialect_path.open('a', newline='') as stream:
                csv.DictWriter(stream, fieldnames=fields).writerows(additions)
        shutil.copyfile(output / f'{STEM}.csv', ROOT / f'data/other/forms/{STEM}.csv')
        shutil.copyfile(HERE / 'source.yaml', ROOT / f'data/other/forms/{STEM}.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'preview')
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    generate(args.output)
    if args.install:
        install()
    print('90 rows; 84 audited cells; 20 sampled cells; ' + ('installed source inputs' if args.install else 'no installation performed'))
