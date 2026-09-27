"""Reproducible stratified source-unit sample, with frozen-hash verification.

Run from any directory: python sample_full.py --seed NUMBER --exclude REPORT ...
Prints JSON to stdout; redirect inside the Jambu workspace to retain selection.
Each --exclude accepts a prior selection/report containing rows or sampled_units.
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
P = Path(__file__).resolve().parent


def sample(seed, exclude_paths, per_stratum=5):
    freeze = json.loads((P/'full-source-freeze-20260926.json').read_text())
    names = ['proposal.csv', 'proposal-audit.jsonl', 'proposal-profile.txt']
    hashes = {n: hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
    assert all(hashes[n] == freeze['sha256'][n] for n in names), 'Frozen proposal changed'
    rows = {r[10]: r for r in csv.reader((P/'proposal.csv').open())}
    units = [json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
    excluded = {f'grierson1908malvirangri:p307:rangri:item:{n}' for n in [32,33,34,35,44]}
    reports = []
    def collect(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {'source_unit_key','roman_source_unit_key','entry_key'} and isinstance(v,str): excluded.add(v)
                elif k in {'entry_keys','pre_reuse_entry_keys'} and isinstance(v,list): excluded.update(v)
                elif k == 'actual_csv_rows': excluded.update(r[10] for r in v)
                else: collect(v)
        elif isinstance(value,list):
            for v in value: collect(v)
    for path in exclude_paths:
        f = Path(path); f = f if f.is_absolute() else P/f
        collect(json.loads(f.read_text()))
        reports.append({'path': f.name, 'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    # A previously sampled occurrence also excludes all aliases of its same output.
    for u in units:
        if u['source_unit_key'] in excluded: excluded.update(u['entry_keys'])
    rng=random.Random(seed); selected=[]; pool_sizes={}
    for stratum in ['grammar','table','specimenI','specimenII']:
        pool=[]
        for u in units:
            match = (stratum=='grammar' and ':grammar:' in u['source_unit_key']) or (stratum=='table' and 'prompt_number' in u) or u.get('section')==stratum
            if match and u['entry_keys'] and u['source_unit_key'] not in excluded and not set(u['entry_keys'])&excluded:
                pool.append(u)
        pool_sizes[stratum]=len(pool)
        for u in rng.sample(pool, per_stratum):
            selected.append({**u,'audit_stratum':stratum,'actual_csv_rows':[rows[k] for k in u['entry_keys']]})
    return {'status':'selected_pending_independent_original_review','seed':seed,'sample_size':len(selected),'hashes':hashes,'exclusion_reports':reports,'excluded_keys':sorted(excluded),'pool_sizes':pool_sizes,'rows':selected}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--exclude',action='append',default=[])
    parser.add_argument('--per-stratum',type=int,default=5)
    args=parser.parse_args()
    print(json.dumps(sample(args.seed,args.exclude,args.per_stratum),ensure_ascii=False,indent=2))
