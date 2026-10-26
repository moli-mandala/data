"""Freeze review inputs and prepare a reproducible, stratified output packet."""
import csv,hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'proposal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=[json.loads(s) for s in (OUT/'audit.jsonl').read_text().splitlines()]
rows={r[10]:r for name in ['wordlist','grammar','native','roman'] for r in csv.reader((OUT/f'{name}.csv').open())}
assert len(rows)==2465
input_names=['wordlist-reviewed.jsonl','morphology-reviewed.jsonl','narrative-first-reading.jsonl','narrative-owner-original-review-20260927.json','orthography-p75-reviewed.json','prose-reviewed.jsonl','prose-owner-review-20260927.json','whole-page-decisions.json','shared-metadata-plan.json','proposed-dialects.csv','build_whole_proposal.py','prepare_proposal_profiles.py','revision2-affix-reconciliation.json']
files={name:sha(HERE/name) for name in input_names}
for p in sorted(OUT.iterdir()):
 if p.is_file() and p.name!='summary.json':files[str(p.relative_to(HERE))]=sha(p)
manifest={'status':'frozen for independent whole-output review; not installation approval','source_pdf_sha256':'aedf47151ebea80852cb7ed66d085fc18310aa3f4d35e189c698b624f3cbef5a','rows':len(rows),'audit_units':len(audit),'legacy_keys_preserved':1336,'files':files,'deferred':['independent output approval','canonical installation and registered-route verification','full CLDF/database build','compiled graph/reference verification','browser QA']}
(HERE/'frozen-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
rng=random.Random(20260927222)
def key(r):return r.get('source_unit_key',r.get('key'))
previous=json.loads((HERE/'revision1/independent-sample-20260927.json').read_text())
excluded={key(x['source_unit']) for x in previous['sample_units']}
emitted=[r for r in audit if r.get('entry_keys') and key(r) not in excluded]
groups=[('wordlist',[r for r in emitted if r.get('prompt')],8),('morphology',[r for r in emitted if r.get('kind')],4),('narrative',[r for r in emitted if ':narrative:' in key(r)],6),('prose',[r for r in emitted if ':prose:' in key(r)],2)]
selected=[]
for label,pool,n in groups:
 for r in rng.sample(pool,n):selected.append({'stratum':label,'source_unit':r,'actual_rows':[rows[k] for k in r['entry_keys']]})
controls=[r for r in audit if not r.get('entry_keys')]
packet={'seed':20260927222,'manifest_sha256':sha(HERE/'frozen-manifest.json'),'sample_units':selected,'excluded_prior_keys':sorted(excluded),'additional_controls':rng.sample(controls,5),'audit_schema':'Each source unit retains physical locator and entry_keys. Several exact repeated units may map to one row; paired segment35 groups2–4 deliberately map to their combined expression. Morphology uses key, others source_unit_key. Original printed page maps to PDF page+8. CSV order: Language,Parameter,Form,Gloss,Native,Phonemic,Notes,Source,unused,Derivation,Entry_Key,Variant_Of,unused,unused,Tags.'}
(HERE/'independent-sample-20260927.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(HERE/'frozen-manifest.json'),'rows':len(rows),'audit_units':len(audit),'sample_units':len(selected)},indent=2))
