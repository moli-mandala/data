"""Deterministic actual-output sample; records no acceptance judgments."""
import csv
import hashlib
import json
import random
from pathlib import Path
P=Path(__file__).resolve().parent
F=P/'whole-proposal-r2'
manifest=json.loads((F/'review-manifest.json').read_text())
for name,digest in manifest['output_sha256'].items():
 assert hashlib.sha256((F/name).read_bytes()).hexdigest()==digest
rows={r[10]:r for r in csv.reader((F/'whole-proposed.csv').open())}
units=list(map(json.loads,(F/'whole-proposed-audit.jsonl').open()))
prior=json.loads((P/'root-independent-selection-2026092701.json').read_text())['records']
excluded={u['source_unit_key'] for r in prior for u in r['audit_records']}
by_entry={}
for u in units:
 for key in u['emitted_entry_keys']:by_entry.setdefault(key,[]).append(u)
eligible={key:us for key,us in by_entry.items() if not any(u['source_unit_key'] in excluded for u in us)}
strata={
 'published_dictionary':sorted(k for k,us in eligible.items() if any(u['representation']=='published PDF' for u in us)),
 'repository_lexicon':sorted(k for k,us in eligible.items() if any(u['representation']=='author repository CSV' and u['source_record'].get('source_file')=='lexicon.csv' for u in us)),
 'translated_examples':sorted(k for k,us in eligible.items() if any(u['source_record'].get('source_file')=='sentences.csv' for u in us)),
 'bound_morphology':sorted(k for k,us in eligible.items() if any(u['representation']=='explicit bound morphology' for u in us)),
}
rng=random.Random(2026092702);selected=[]
for stratum,count in [('published_dictionary',8),('repository_lexicon',6),('translated_examples',4),('bound_morphology',2)]:
 pool=[k for k in strata[stratum] if k not in {r['entry_key'] for r in selected}]
 for key in rng.sample(pool,count):
  selected.append(dict(sample=len(selected)+1,stratum=stratum,entry_key=key,row=rows[key],audit_records=by_entry[key]))
assert len(selected)==20
controls=[]
specs=[
 ('printed-number conflict','kholosi:pdf:p1:c1:u24'),
 ('probable competing ancestry','kholosi:pdf:p8:c1:u7'),
 ('printed-number conflict','kholosi:pdf:p8:c1:u19'),
 ('explicit local compound','derāz kadden'),
 ('rejected ancestry','xallo'),
 ('qualified comparison scope','bere'),
 ('explicit optional form','kholosi:github:18b43f982efa:lexicon.csv:row141'),
 ('pronunciation alternatives','kholosi:github:18b43f982efa:lexicon.csv:row44'),
 ('head and feminine paradigm','kholosi:github:18b43f982efa:lexicon.csv:row37'),
 ('untranslated response','kholosi:github:18b43f982efa:sentences.csv:row91'),
 ('source blank','kholosi:github:18b43f982efa:sentences.csv:row95'),
 ('untranslated whole text','kholosi:github:18b43f982efa:sentences.txt:whole-text'),
 ('acoustic control','kholosi:github:18b43f982efa:data.txt'),
 ('mixed-case citation','kholosi:github:18b43f982efa:lexicon.csv:row137'),
]
for purpose,locator in specs:
 matches=[u for u in units if u['source_unit_key']==locator or (u['representation']=='published PDF' and u['source_record'].get('form')==locator)]
 assert matches,(purpose,locator)
 controls.append(dict(purpose=purpose,audit_records=matches,
  actual_rows=[rows[k] for u in matches for k in u['emitted_entry_keys']]))
out=dict(seed=2026092702,status='selection only; independent reviewer must inspect originals',
 method='Stratified random actual emitted rows, excluding every source unit represented in r1 sample; controls are separate and may revisit earlier failures.',
 excluded_source_keys=sorted(excluded),manifest=manifest,records=selected,targeted_controls=controls)
(P/'independent-selection-r2-2026092702.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Selected20 fresh actual rows plus',len(controls),'targeted units/groups')
