"""Reproduce the complete chapter preview without installing or building a DB."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = 'grierson1906lsi4'
SITES = {
    site: f'dialect:Turi:turi_{site.lower()}_lsi1906:{site}'
    for site in ('Sambalpur', 'Jashpur', 'Ranchi', 'Sarangarh')
}


def read_units():
    files = ['prose-recovery-review.jsonl'] + [
        f'specimen{n}-first-reading.jsonl' for n in range(1, 4)
    ]
    units = []
    for name in files:
        units.extend(json.loads(line) for line in (ROOT / name).read_text().splitlines())
    assert len(units) == 352
    assert len({r['entry_key'] for r in units}) == len(units)
    return units, files


def build():
    units, files = read_units()
    rows, audit, identities = [], [], {}
    for unit in units:
        record = dict(unit)
        if unit['status'] in ('contextual_comparator', 'editorial_insertion_not_independent_attestation'):
            record['preview_disposition'] = 'context_only'
            record['output_entry_key'] = None
            audit.append(record)
            continue
        form = unit['form_first_reading']
        gloss = unit.get('source_aligned_gloss', unit.get('gloss'))
        assert form and gloss
        site = unit['site']
        page = unit['printed_page']
        locator = (f"prose example {unit['item']}" if 'item' in unit else
                   f"specimen line {unit['line']}, unit {unit['position']}")
        citation = f'{SOURCE}[p. {page}, {site}, {locator}]'
        identity = (site, form.casefold(), gloss.casefold())
        source_note = unit.get('source_note', '')
        uncertainty = unit.get('typed_uncertainty', '')
        notes = source_note
        if uncertainty:
            notes = ' '.join(filter(None, [notes, f'Reading uncertain: {uncertainty}']))
        if identity in identities:
            row = rows[identities[identity]]
            row[7] += '; ' + citation
            if notes and notes not in row[6]:
                row[6] = ' '.join(filter(None, [row[6], notes]))
            if uncertainty and 'uncertain' not in row[14].split():
                row[14] = ' '.join(filter(None, [row[14], 'uncertain']))
            record['preview_disposition'] = 'attestation_reuse'
        else:
            tags = [SITES[site]] if site in SITES else []
            tags.extend(unit.get('grammar_tags', []))
            if ' ' in form:
                tags.append('multiword-expression')
            if uncertainty:
                tags.append('uncertain')
            row = ['Turi', '', form, gloss, '', '', notes, citation, '', '',
                   unit['entry_key'], '', '', '', ' '.join(tags)]
            identities[identity] = len(rows)
            rows.append(row)
            record['preview_disposition'] = 'candidate_row'
        record['output_entry_key'] = row[10]
        audit.append(record)
    legacy = [json.loads(line) for line in (ROOT / 'audit.jsonl').read_text().splitlines()
              if json.loads(line)['status'] == 'ingested']
    keys = {row[10] for row in rows}
    assert all(unit['entry_key'] in keys for unit in legacy)
    return rows, audit


def prepare():
    units, files = read_units()
    rows, audit = build()
    legacy = [json.loads(line) for line in (ROOT / 'audit.jsonl').read_text().splitlines()
              if json.loads(line)['status'] == 'ingested']
    target = ROOT / 'full-review.csv'
    with target.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (ROOT / 'full-review-audit.jsonl').write_text(''.join(
        json.dumps(record, ensure_ascii=False) + '\n' for record in audit))
    report = {
        'status': 'review_preview_not_installed',
        'source_units': len(units), 'candidate_rows': len(rows),
        'dispositions': dict(Counter(r['preview_disposition'] for r in audit)),
        'site_rows': dict(Counter(r['site'] for r in audit if r['preview_disposition'] == 'candidate_row')),
        'uncertain_units': [r['entry_key'] for r in units if r.get('typed_uncertainty')],
        'legacy_keys_preserved': len(legacy),
        'csv_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'input_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in files},
        'pending': ['Independent full-output audit', 'Sound profile and Sarangarh registry',
                    'Source metadata and focused validation', 'Canonical installation'],
        'deferred_by_user': ['Full data build', 'Database and browser refresh'],
    }
    (ROOT / 'full-review-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    prepare()
