import csv, hashlib, json, random
from pathlib import Path
H=Path(__file__).resolve().parent
I=H.parent/'integration-proposal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((H/'selection.json').read_text())
excluded={r['source_unit_key'] for field in ('selected_rows','targeted_rows') for r in prior[field]}
audit=[json.loads(x) for x in (I/'comparison-routing-audit.jsonl').read_text().splitlines()]
csvrows=list(csv.reader((I/'comparison-rows.csv').open()))
bykey={r[10]:r for r in csvrows}
rng=random.Random(20260927502); selected=[]; pages=set()
for code,n in [('A',4),('B',3),('J',4),('U',3),('V',3),('k',3)]:
    pool=[r for r in audit if r['source_code']==code and r['source_unit_key'] not in excluded]
    for _ in range(n):
        fresh=[r for r in pool if r['historical_audit_record']['PDF_Page'] not in pages]
        r=rng.choice(fresh or pool);pool.remove(r);selected.append(r);pages.add(r['historical_audit_record']['PDF_Page'])
        assert r['proposed_csv_row']==bykey[r['source_unit_key']]
out=dict(seed=20260927502,excluded_keys=sorted(excluded),input_hashes={f:sha(I/f) for f in ('comparison-rows.csv','comparison-routing-audit.jsonl','comparison-dialects.csv','comparison-routing-status.json')},selected_rows=selected)
(H/'routed-selection.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for n,r in enumerate(selected,1): print(n,r['historical_audit_record']['PDF_Page'],r['item'],r['source_code'],r['form'],r['proposed_csv_row'][3],r['proposed_csv_row'][14])
