"""Fresh stratified sample excluding both failed random-sample source sets."""
import csv
import hashlib
import json
import random
from pathlib import Path
P=Path(__file__).resolve().parent;F=P/'whole-proposal-r3'
manifest=json.loads((F/'review-manifest.json').read_text())
for name,digest in manifest['output_sha256'].items():assert hashlib.sha256((F/name).read_bytes()).hexdigest()==digest
rows={r[10]:r for r in csv.reader((F/'whole-proposed.csv').open())}
units=list(map(json.loads,(F/'whole-proposed-audit.jsonl').open()))
excluded=set()
for name in ('root-independent-selection-2026092701.json','independent-selection-r2-2026092702.json'):
 for r in json.loads((P/name).read_text())['records']:
  excluded.update(u['source_unit_key'] for u in r['audit_records'])
by_entry={}
for u in units:
 for k in u['emitted_entry_keys']:by_entry.setdefault(k,[]).append(u)
eligible={k:us for k,us in by_entry.items() if not any(u['source_unit_key'] in excluded for u in us)}
predicates={
 'published_dictionary':lambda u:u['representation']=='published PDF',
 'repository_lexicon':lambda u:u['representation']=='author repository CSV' and u['source_record'].get('source_file')=='lexicon.csv',
 'translated_examples':lambda u:u['source_record'].get('source_file')=='sentences.csv',
 'bound_morphology':lambda u:u['representation']=='explicit bound morphology',
}
rng=random.Random(2026092703);selected=[]
for stratum,count in [('published_dictionary',8),('repository_lexicon',6),('translated_examples',4),('bound_morphology',2)]:
 pool=sorted(k for k,us in eligible.items() if any(predicates[stratum](u) for u in us) and k not in {r['entry_key'] for r in selected})
 for k in rng.sample(pool,count):selected.append(dict(sample=len(selected)+1,stratum=stratum,entry_key=k,row=rows[k],audit_records=by_entry[k]))
controls=[]
for n in (92,194,141,169,64):
 k=f'legacy:20220913-kholosi:row:{n}'
 controls.append(dict(purpose='independently supported historical-number correction',entry_key=k,row=rows[k],audit_records=by_entry[k]))
for locator in ('kholosi:pdf:p4:c2:u7','kholosi:pdf:p10:c1:u11','kholosi:pdf:p6:c1:u21','kholosi:pdf:p6:c1:u22','kholosi:pdf:p6:c1:u23',
                'kholosi:github:18b43f982efa:lexicon.csv:row71','kholosi:github:18b43f982efa:lexicon.csv:row153'):
 u=next(u for u in units if u['source_unit_key']==locator)
 controls.append(dict(purpose='donor attribution versus component/variant scope',audit_records=[u],actual_rows=[rows[k] for k in u['emitted_entry_keys']]))
result=dict(seed=2026092703,status='selection only; no acceptance judgment',manifest=manifest,
 method='20 stratified actual rows excluding all source units represented in r1/r2 random samples;12 targeted correction controls separately revisited.',
 excluded_source_keys=sorted(excluded),records=selected,targeted_controls=controls)
(P/'independent-selection-r3-2026092703.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Fresh20 plus',len(controls),'targeted controls')
