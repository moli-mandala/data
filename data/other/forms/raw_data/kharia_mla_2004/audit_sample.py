"""Select reproducible full-snapshot raw/output samples; verify a retained manual audit."""
import argparse
import hashlib
import json
import random
from import_source import HERE, CSV, prepare


def sample(seed, count=20, exclude=()):
    rows, audit = prepare()
    by_key = {r[10]: r for r in rows}
    population = [a for a in audit if a['status'] == 'ingested' and a['entry_key'] not in exclude]
    return [dict(a, installed_rows=[by_key[r['entry_key']] for r in a['rows'] + a.get('supplementary_rows', [])])
            for a in random.Random(seed).sample(population, count)]


def verify(report_path):
    report = json.loads(report_path.read_text())
    assert report.get('material_error_records', report.get('material_errors')) == 0
    assert len(report['entries']) == report['sample_size'] == 20
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == report['staged_csv_sha256']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2026092621)
    parser.add_argument('--exclude-report', action='append', default=[])
    parser.add_argument('--verify', type=str)
    args = parser.parse_args()
    if args.verify:
        verify(HERE / args.verify)
        print('Retained independent 20-entry audit matches canonical source CSV.')
    else:
        excluded = set()
        for name in args.exclude_report:
            loaded = json.loads((HERE / name).read_text())
            entries = loaded if isinstance(loaded, list) else loaded['entries']
            excluded.update(entry['entry_key'] for entry in entries)
        print(json.dumps(sample(args.seed, exclude=excluded), ensure_ascii=False, indent=2))
