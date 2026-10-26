"""Freeze a seeded actual-output selection before original-image review."""
import csv, hashlib, json, random
from pathlib import Path
here=Path(__file__).resolve().parent
assembly=here.parents[1]/'integration/review-assembly-r1'
out=here/'fresh20-selection-20260927510.json'
assert not out.exists(), 'Never overwrite a frozen selection'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=[json.loads(x) for x in (assembly/'combined-review-audit.jsonl').read_text().splitlines()]
actual={r[10]:r for r in csv.reader((assembly/'combined-review-candidate.csv').open())}
seed=20260927510
rng=random.Random(seed)
strata={'grammar':4,'No37-Roman-lexical-emission':5,'No37-whole-source-expression':1,'No38-atomic':4,'No38-expression':1,'quoted-commentary':2,'outside-eligible':3}
samples=[]
for kind,n in strata.items():
    population=sorted([x for x in audit if x['kind']==kind],key=lambda x:x['source_unit_key'])
    for x in rng.sample(population,n):
        assert actual[x['source_unit_key']]==x['actual_row']
        samples.append({'sample':len(samples)+1,'key':x['source_unit_key'],'kind':kind,'row':x['actual_row'],'source_record':x['source_record']})
out.write_text(json.dumps({'seed':seed,'selection_frozen_before_original_review':True,'csv_sha256':sha(assembly/'combined-review-candidate.csv'),'audit_sha256':sha(assembly/'combined-review-audit.jsonl'),'strata':strata,'count':20,'samples':samples},ensure_ascii=False,indent=2)+'\n')
print('selection_sha256',sha(out))
for x in samples:print(x['sample'],x['kind'],x['key'],repr(x['row'][2]),repr(x['row'][3]))
