from pathlib import Path
import csv,json,random,hashlib
H=Path(__file__).resolve().parent;P=H.parent
csvpath=P/'integration/review-candidate.csv';rows=list(csv.reader(csvpath.open()));rng=random.Random(20260927401)
groups={'comparisons':([],6),'pronouns_and_expressions':([],4),'morphology':([],4),'paradigm':([],4),'vocabulary':([],2)}
for r in rows:
 k=r[10]
 group='vocabulary' if ':vocabulary:' in k else 'paradigm' if ':verb-paradigm:' in k else 'comparisons' if ':p9:' in k or ':geminate-consonant:' in k else 'morphology' if any(s in k for s in [':plural-prose:',':dative-prose:',':tense-suffixes:']) else 'pronouns_and_expressions'
 groups[group][0].append(r)
samples=[]
for name,(candidates,n) in groups.items():
 for r in rng.sample(sorted(candidates,key=lambda r:r[10]),n):samples.append(dict(stratum=name,key=r[10],row=r))
x=dict(seed=20260927401,method='Seeded stratified sample of actual frozen output; 6 comparison, 4 pronoun/expression, 4 morphology, 4 paradigm, 2 vocabulary rows.',csv_sha256=hashlib.sha256(csvpath.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256((P/'integration/review-audit.jsonl').read_bytes()).hexdigest(),samples=samples)
out=H/'fresh-original-selection.json';text=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
if out.exists():assert out.read_text()==text
else:out.write_text(text)
for i,s in enumerate(samples,1):print(i,s['key'],s['row'][0],repr(s['row'][2]),repr(s['row'][3]),s['row'][14])
