"""Generate rich CSV and per-record audit; installation follows acceptance review."""
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from proposal import build as proposal
from transcription import SPACING

HERE = Path(__file__).resolve().parent
STEM = '20260921-dhakal-darai'
SOURCE = 'dhakal2011darai'
ROOT = HERE.parents[4]


def build():
    data = proposal()
    inventory = {r['key']: r for r in json.loads((HERE / 'inventory.json').read_text())}
    exceptions = {r['key']: r for r in json.loads((HERE / 'inventory-exceptions.json').read_text())}
    metadata = json.loads((HERE / 'metadata-review.json').read_text())
    dialects = {d['ID']: d['Tag'] for d in metadata['dialects']}
    rows, audit = [], []
    for entry in data['candidates']:
        page, key = entry['printed_page'], entry['key']
        raw = inventory.get(key, exceptions.get(key))
        tags = [*entry['tags'], dialects['darai_chitwan']]
        if page == 49:
            tags.append(dialects['darai_pidrahani'])
        phonemic = entry['form']
        if key in SPACING:
            before, phonemic = SPACING[key]
            assert entry['form'] == before
        locator = f'p. {page}'
        if key == 'dhakal2011:p74:g621':
            locator += ', example 67c, text KaQ.SLD.079'
        if key == 'dhakal2011:p74:g697':
            locator += ', example 67d, text DP.CND.065'
        row = ['Darai', '', entry['form'], entry['gloss'], '', phonemic, '',
               f'{SOURCE}[{locator}]', '', entry.get('source_analysis', ''), key,
               entry.get('variant_of_key', ''), '', '', ' '.join(tags)]
        rows.append(row)
        audit.append({'key': key, 'status': 'proposed', 'printed_page': page,
                      'pdf_page': page + 22, 'raw': raw, 'parsed': entry,
                      'output': row, 'dialect_policy': metadata['mapping'],
                      'consultant_provenance': metadata['consultant_provenance'] if page == 49 else None})
    for excluded in data['excluded']:
        raw = inventory[excluded['key']]
        audit.append({'key': excluded['key'], 'status': 'excluded',
                      'printed_page': raw['printed_page'],
                      'pdf_page': raw['printed_page'] + 22,
                      'raw': raw, 'decision': excluded, 'output': None})
    assert len(rows) == 485 and len(audit) == 502
    assert len({r[10] for r in rows}) == len(rows)
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    return rows, audit


def generate():
    output = HERE / 'preview'
    output.mkdir(exist_ok=True)
    rows, audit = build()
    path = output / f'{STEM}.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    seed = 2026092107
    # Randomize across the complete raw-record population, including exclusions.
    sample = random.Random(seed).sample(audit, 20)
    manifest = {'seed': seed, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'audit_sha256': hashlib.sha256((output / 'audit.json').read_bytes()).hexdigest(),
                'records': sample, 'status': 'awaiting source-to-output review'}
    (output / 'acceptance-sample.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(rows)} rich rows; {len(audit)} audit records; seeded review awaits acceptance')
    return manifest


def install():
    from pybtex.database import parse_file
    sample = generate()
    acceptance = json.loads((HERE / 'acceptance-results.json').read_text())
    if acceptance['material_errors'] != 0 or acceptance['sample'] != sample:
        raise ValueError('Current rich output is not covered by acceptance review')
    for relative, digest in acceptance['input_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reviewed input changed: {relative}')
    bib = ROOT / 'cldf/sources.bib'
    existing = parse_file(str(bib)).entries
    proposed = parse_file(str(HERE / 'source.bib')).entries
    for key, entry in proposed.items():
        if key in existing and existing[key] != entry:
            raise ValueError(f'Bibliography conflict: {key}')
    metadata = json.loads((HERE / 'metadata-review.json').read_text())
    dialect_path = ROOT / 'cldf/dialects.csv'
    with dialect_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, dialect_rows = reader.fieldnames, list(reader)
    dialects = {r['ID']: r for r in dialect_rows}
    for dialect in metadata['dialects']:
        if dialect['ID'] in dialects and dialects[dialect['ID']] != dialect:
            raise ValueError(f'Dialect conflict: {dialect["ID"]}')
    language_path = ROOT / 'cldf/languages.csv'
    with language_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        language_fields, languages = reader.fieldnames, list(reader)
    language = next(r for r in languages if r['ID'] == 'Darai')
    assert language['Glottocode'] == 'dara1250' and language['Clade'] in {'Other', 'Bihari'}
    # All conflict checks above precede mutations; do not replace unrelated rows.
    with bib.open('a') as stream:
        blocks = re.split(r'(?=^@)', (HERE / 'source.bib').read_text(), flags=re.M)
        for key in proposed:
            if key not in existing:
                block = next(b for b in blocks if re.match(r'@\w+\{' + re.escape(key) + ',', b))
                stream.write('\n' + block.rstrip() + '\n')
    with dialect_path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        for dialect in metadata['dialects']:
            if dialect['ID'] not in dialects:
                writer.writerow(dialect)
    if language['Clade'] == 'Other':
        language['Clade'] = 'Bihari'
        with language_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=language_fields)
            writer.writeheader()
            writer.writerows(languages)
    for extension in ['csv', 'yaml']:
        source = HERE / ('preview' if extension == 'csv' else '') / f'{STEM}.{extension}'
        shutil.copyfile(source, ROOT / f'data/other/forms/{STEM}.{extension}')
    print('Installed 485 source rows and reviewed metadata; full integration pending')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    install() if args.install else generate()
