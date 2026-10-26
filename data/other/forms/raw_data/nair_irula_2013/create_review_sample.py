"""Reproducible disjoint physical-unit sample, pinned to current source outputs."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
HERE=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(seed,revision):
    manifest=json.loads((HERE/f'review-manifest-{revision}.json').read_text())
    assert all(SHA(HERE/n)==v for n,v in manifest['sha256'].items()),'frozen input drift'
    audit=[json.loads(s) for s in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
    rows={r[10]:r for r in csv.reader((HERE/'proposed.csv').open())}
    prior_files=sorted(HERE.glob('independent-selection-*.json'))
    excluded={s['unit_key'] for path in prior_files for s in json.loads(path.read_text()).get('sample',[])}
    rng=random.Random(seed);sample=[]
    for page,count in [(22,3),(23,3),(24,4),(25,4),(26,3),(27,3)]:
        candidates=[a for a in audit if a['source_record']['printed_page']==page and a['proposed_entry_keys'] and a['unit_key'] not in excluded]
        for a in rng.sample(candidates,count):
            key=rng.choice(a['proposed_entry_keys'])
            sample.append({'unit_key':a['unit_key'],'entry_key':key,'csv_row':rows[key],
                           'source_record':a['source_record'],'typed_uncertainties':a['typed_uncertainties'],
                           'original_image':f'original-review/page-{page}.png'})
    assert len(sample)==len({s['unit_key'] for s in sample})==20
    controls=[a for a in audit if a['typed_uncertainties'] or a['disposition']=='audit-only' or any(e.get('derivation_parent_keys') or e.get('variant_of_key') for e in a['proposed_emissions'])]
    result={'seed':seed,'csv_sha256':SHA(HERE/'proposed.csv'),'audit_sha256':SHA(HERE/'proposal-audit.jsonl'),
            'excluded_prior_unit_keys':sorted(excluded),'prior_selection_files':[p.name for p in prior_files],
            'sampled_count':20,'sample':sample,'targeted_controls':controls}
    assert not excluded & {s['unit_key'] for s in sample}
    dest=HERE/f'independent-selection-{revision}-{seed}.json';assert not dest.exists(),'sample immutable'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(dest)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--revision',default='r1');args=parser.parse_args();main(args.seed,args.revision)
