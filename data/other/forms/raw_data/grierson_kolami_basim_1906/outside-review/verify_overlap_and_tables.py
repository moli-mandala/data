"""Bounded exact canonical overlap and cached table discovery, no mutation."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/source_meta.py').exists())
RAW = ROOT / 'data/data/other/forms/raw_data'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.casefold()) if not unicodedata.combining(c))

def main():
    files = {name: ROOT / f'data/data/other/forms/20260927-grierson-{name}.csv' for name in ('kui', 'naiki')}
    rows = {name: {r[10]: r for r in csv.reader(path.open(newline=''))} for name, path in files.items()}
    exact = [
        ('kui', 'grierson1906kui:p566:outside-comparand:06', 'Kolami', 'and', 'he', '566'),
        ('kui', 'grierson1906kui:p623:outside-comparand:15', 'Kolami', 'nē', '', '623'),
        ('naiki', 'grierson1906naiki:p623:genitive-comparison:naiki', 'Naiki', 'nē', '', '623'),
    ]
    checks = []
    for source, key, language, form, gloss, page in exact:
        row = rows[source][key]
        assert row[0] == language and row[2] == form and row[3] == gloss
        assert f'p. {page}' in row[7]
        checks.append({'source': source, 'key': key, 'canonical_row': row, 'same_physical_occurrence': True, 'key_reuse_not_new_attestation': True,
                       'literal_reuse_accepted': page != '566',
                       'literal_caveat': 'Current canonical and vs new grammar amd awaits sharp-original adjudication. Physical-key identity only, not accepted spelling equality; preserve one occurrence with correction history.' if page == '566' else 'Exact nē original/canonical comparison remains accepted.'})
    held = [f'grierson1906kui:p566:outside-comparand:{n:02}' for n in range(1, 6)]
    assert not any(key in rows['kui'] for key in held)
    held_source = RAW / 'grierson_kui_1906/outside-comparands-reviewed-candidate-20260927.jsonl'
    held_records = [json.loads(line) for line in held_source.open() if json.loads(line).get('source_unit_key') in held]
    assert [r['source_unit_key'] for r in held_records] == held
    table = RAW / 'grierson_holiya_1906/nordic-vol4-part1-tables.html'
    pieces = re.split(r'<div id="(\d+)"[^>]*>', table.read_text())
    matches = []
    for i in range(1, len(pieces), 2):
        page, block = pieces[i], pieces[i + 1]
        for index, raw in enumerate(re.findall(r'<tr>(.*?)</tr>', block, re.S), 1):
            cells = [html.unescape(re.sub(r'<[^>]*>', '', c)).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>', raw, re.S)]
            if any(re.search(r'\b(?:kolam\w*|colam\w*|basim|pusad)\b', fold(c)) for c in cells):
                matches.append({'cached_page_id': int(page), 'original_folio_not_independently_verified_here': True, 'row_index_in_cached_block': index, 'cells': cells, 'raw_row_html': raw, 'disposition': 'population_or_distribution_table_not_lexical'})
    report = {
        'status': 'Canonical physical-key checks pass; p623 literal reuse closed, p566 and/amd literal equality pending; table discovery classified',
        'canonical_sha256': {name: sha(path) for name, path in files.items()},
        'canonical_counts': {name: len(records) for name, records in rows.items()},
        'exact_emitted_occurrences': checks,
        'counts': {'canonical_rows_with_physical_overlap': 3, 'literal_reuse_rows_closed': 2, 'literal_overlap_rows_pending': 1, 'physical_printings_represented': 2, 'of_which_outside_chapter_printings': 1},
        'historical_basim_held_records': held_records,
        'held_record_source_sha256': sha(held_source),
        'historical_hold_policy': 'Five p566 records remain absent from current canonical Kui CSV and retained in audit. Later Naikri identity reconciliation is a distinct decision, not deletion of history or an invented Form_ID alias.',
        'cached_table_sha256': sha(table),
        'table_label_matches': matches,
        'table_limits': 'No lexical Kolami/Basim table identified in cached HTML; original complete standard-list header evidence is separately required and reused. Cached representation is not proof of absent material from an original.',
        'canonical_mutations': False,
    }
    (HERE / 'exact-overlap-and-table-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'canonical_counts': report['canonical_counts'], 'reuse_counts': report['counts'], 'table_label_pages': [r['cached_page_id'] for r in matches], 'historical_holds': len(held_records)}))

if __name__ == '__main__':
    main()
