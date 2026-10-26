"""Reproducible disjoint physical-unit sample, pinned to current source outputs."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
HERE=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(seed):
    manifest=json.loads((HERE/'review-manifest-r1.json').read_text())
    assert all(SHA(HERE/n)==v for n,v in manifest['sha256'].items()),'frozen input drift'
    audit=[json.loads(s) for s in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
    rows={r[10]:r for r in csv.reader((HERE/'proposed.csv').open())}
    rng=random.Random(seed);sample=[]
    for page,count in [(22,3),(23,3),(24,4),(25,4),(26,3),(27,3)]:
        candidates=[a for a in audit if a['source_record']['printed_page']==page and a['proposed_entry_keys']]
        for a in rng.sample(candidates,count):
            key=rng.choice(a['proposed_entry_keys'])
            sample.append({'unit_key':a['unit_key'],'entry_key':key,'csv_row':rows[key],
                           'source_record':a['source_record'],'typed_uncertainties':a['typed_uncertainties'],
                           'original_image':f'original-review/page-{page}.png'})
    assert len(sample)==len({s['unit_key'] for s in sample})==20
    controls=[a for a in audit if a['typed_uncertainties'] or a['disposition']=='audit-only' or any(e.get('derivation_parent_keys') or e.get('variant_of_key') for e in a['proposed_emissions'])]
    result={'seed':seed,'csv_sha256':SHA(HERE/'proposed.csv'),'audit_sha256':SHA(HERE/'proposal-audit.jsonl'),
            'excluded_prior_unit_keys':[],'exclusion_reason':'No earlier Irula independent sample exists; Betta/other chapters have different source keys.',
            'sampled_count':20,'sample':sample,'targeted_controls':controls}
    dest=HERE/f'independent-selection-r1-{seed}.json';assert not dest.exists(),'sample immutable'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(dest)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);main(parser.parse_args().seed)
