"""Sample actual lightweight parser output; does not run a data/database build."""
import argparse
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[4]))
from make_cldf import parse_file
from import_source import STEM, prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    expected, audit = prepare()
    raw = {r[10]: r for r in expected}
    cells = {r[10]: a['source_cell'] for a in audit for r in a['emitted_rows']}
    errors = io.StringIO()
    parsed, stats = parse_file(args.csv, errors=errors, name=STEM)
    assert not errors.getvalue(), errors.getvalue()
    assert len(parsed) == len(raw) == 441
    assert {r.entry_key for r in parsed} == set(raw)
    sample = random.Random(args.seed).sample(sorted(parsed, key=lambda r:r.entry_key), 20)
    result = {'seed': args.seed, 'parser_stats': stats, 'sample': []}
    for row in sample:
        result['sample'].append({'entry_key': row.entry_key, 'source_cell': cells[row.entry_key], 'raw_csv': raw[row.entry_key], 'parsed': {'language': row.lang, 'form': row.form, 'original': row.old_form, 'gloss': row.gloss, 'tags': row.tags, 'source': row.source}})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for r in result['sample']:
        print(r['entry_key'], r['source_cell']['Raw_Response'], '=>', r['parsed'])
