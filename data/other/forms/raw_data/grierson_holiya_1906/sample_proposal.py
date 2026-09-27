"""Seeded20 whole-source selection for independent original-image review."""
from pathlib import Path
import argparse,csv,hashlib,json,random
P=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',required=True);ap.add_argument('--exclude',action='append',default=[]);args=ap.parse_args()
    excluded=set()
    for filename in args.exclude:
        report=json.loads((P/filename).read_text())
        excluded.update(x['source_unit_key'] for x in report.get('sample',report.get('selected_units',[])))
    units=[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()];rows={r[10]:r for r in csv.reader((P/'proposal.csv').open())};rng=random.Random(args.seed);sample=[]
    for page,count in [(386,2),(387,2)]+[(page,2) for page in range(388,396)]:
        available=[u for u in units if u['printed_page']==page and u['entry_keys'] and u['source_unit_key'] not in excluded]
        for u in rng.sample(available,count):sample.append({**u,'actual_csv_rows':[rows[k]for k in u['entry_keys']],'source_image':f'tmp/holiya-completion/dsal-{page+20:03d}.jpg'})
    assert len(sample)==20
    data={'status':'selected; original-image comparison pending','seed':args.seed,'sample':sample,'hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['proposal.csv','proposal-audit.jsonl','proposal-profile.txt']}}
    destination=P/args.output
    if destination.exists():raise SystemExit('Refusing to replace an existing audit selection')
    destination.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
