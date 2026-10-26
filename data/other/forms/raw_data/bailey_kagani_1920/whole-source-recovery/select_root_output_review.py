"""Seeded, page-stratified review of immutable r1 output, not first readings."""
import argparse, csv, hashlib, json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--revision',default='whole-proposal-r1')
parser.add_argument('--seed',type=int,default=20260927151)
args=parser.parse_args()
REV = HERE / args.revision
manifest = json.loads((REV / 'review-manifest.json').read_text())
for name, expected in manifest['output_sha256'].items():
    assert hashlib.sha256((REV / name).read_bytes()).hexdigest() == expected, name
rows = {r[10]: r for r in csv.reader((REV / 'proposal.csv').open())}
audit = [json.loads(s) for s in (REV / 'proposal-audit.jsonl').read_text().splitlines()]
pages = {}
for unit in audit:
    if not unit.get('entry_keys'):
        continue
    page = unit.get('printed_page', unit.get('input', {}).get('printed_page'))
    if page is not None:
        pages.setdefault(int(page), []).append(unit)
rng = random.Random(args.seed)
# Twenty distinct pages; introduction89 and90 share a stratum.
selected_pages = [88, rng.choice([89,90]), *range(92,110)]
assert len(selected_pages) == 20
selected = []
for page in selected_pages:
    unit = rng.choice(pages[page])
    selected.append({'printed_page':page, 'source_unit':unit,
                     'actual_rows':[rows[k] for k in unit['entry_keys']]})
result = {'seed':args.seed, 'manifest_sha256':hashlib.sha256((REV/'review-manifest.json').read_bytes()).hexdigest(),
          'units':len(selected), 'rows':sum(len(s['actual_rows']) for s in selected), 'sample':selected}
target = REV / 'root-output-sample-20260927.json'
payload = json.dumps(result,ensure_ascii=False,indent=2)+'\n'
assert not target.exists() or target.read_text()==payload
target.write_text(payload)
for item in selected:
    for r in item['actual_rows']:
        print(json.dumps({'page':item['printed_page'],'key':r[10],'form':r[2],'gloss':r[3],'notes':r[6],'citation':r[7],'tags':r[14]},ensure_ascii=False))
