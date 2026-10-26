import csv,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent.parent;I=P/'integration';O=Path(__file__).resolve().parent
rows={r[10]:r for r in csv.reader((I/'combined-candidate.csv').open())}
audit=[json.loads(x) for x in (I/'combined-audit.jsonl').read_text().splitlines()]
groups={'table':[],'grammar':[],'specimen':[],'outside':[]}
for a in audit:
 if a['disposition']!='emitted':continue
 sec=a['section'];group='specimen' if sec.startswith('specimen') else 'outside' if sec.startswith('outside') else sec
 if group not in groups:continue
 for key in a['entry_keys']:groups[group].append((key,a))
rng=random.Random(20260927317);sample=[]
for group,pool in groups.items():
 for key,a in rng.sample(sorted(pool,key=lambda x:x[0]),5):sample.append({'entry_key':key,'section':group,'row':rows[key],'audit':a})
report={'seed':20260927317,'design':'Five unique emitted rows sampled without replacement in each of table, grammar, specimens and outside comparisons; all final-output strata covered. No prior random final-output sample exists in this review directory.',
 'frozen_hashes':{n:hashlib.sha256((I/n).read_bytes()).hexdigest() for n in ['combined-candidate.csv','combined-audit.jsonl','literal-preservation.txt','install-settings-proposed.yaml']},'sample':sample}
(O/'independent-selection-20260927317.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for n,s in enumerate(sample,1):print(n,s['entry_key'],repr(s['row'][2]),repr(s['row'][3]))
