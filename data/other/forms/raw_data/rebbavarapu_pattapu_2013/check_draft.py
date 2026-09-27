"""Exercise the draft through the row parser, never the data-build pipeline."""
import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[4]))
import make_cldf
import source_meta
from segments.tokenizer import Tokenizer
from import_source import build

def parse_draft(path):
    meta = source_meta.SourceMeta([ROOT/'20260922-pattapu-iso.yaml'])
    converter = Tokenizer(str(ROOT/'pattapu-iso.txt'))
    errors = io.StringIO()
    with patch.object(source_meta, 'load', return_value=meta), patch.dict(make_cldf.convertors, {'pattapu-iso': converter}):
        parsed, stats = make_cldf.parse_file(str(path), errors, name='20260922-pattapu-iso')
    assert not errors.getvalue(), errors.getvalue()
    return parsed, stats

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    directory = args.output/'other/forms'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'20260922-pattapu-iso.csv'
    rows, audit = build()
    with path.open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    parsed, stats = parse_draft(path)
    assert len(parsed) == len(rows) == 201
    raw = {r[10]: r for r in rows}
    assert set(raw) == {r.entry_key for r in parsed}
    for r in parsed:
        assert r.old_form == raw[r.entry_key][2]
        assert r.gloss == raw[r.entry_key][3]
    sample = random.Random(args.seed).sample(sorted(parsed, key=lambda r:r.entry_key), 20)
    report = {'seed': args.seed, 'stats': stats, 'status': 'sample prepared; manual acceptance review pending', 'sample': [
        {'key': r.entry_key, 'form': r.form, 'original': r.old_form, 'gloss': r.gloss, 'tags': r.tags, 'source': r.source}
        for r in sample]}
    (args.output/f'output-sample-{args.seed}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
