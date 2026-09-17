import json,csv,io,hashlib,os,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;root=P.parents[2];stem='near-pass76';old='Marathi/Marwari kuto';new='Marwari kuto'
overlay=root/'data/etymology-assignments.csv';raw=overlay.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
r=list(csv.DictReader(io.StringIO(raw.decode())));ids={x['Form_ID'] for x in json.loads((P/(stem+'-saved-assignments.json')).read_text()) if old in x['Notes']};assert len(ids)==2
for x in r:
 if x['Form_ID'] in ids:assert old in x['Notes'];x['Notes']=x['Notes'].replace(old,new)
out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(r[0]));w.writeheader();w.writerows(r);payload=out.getvalue().encode()
(P/'backups/etymology-assignments-before-pass76-note-correction.csv').write_bytes(raw)
with tempfile.NamedTemporaryFile(dir=overlay.parent,delete=False) as f:f.write(payload);tmp=Path(f.name)
assert overlay.read_bytes()==raw;os.replace(tmp,overlay)
def fix(x):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return [fix(v) for v in x]
 if isinstance(x,dict):return {k:fix(v) for k,v in x.items()}
 return x
paths=[P/(stem+'-'+s+'.json') for s in ['decisions','rules','saved-assignments']]+[Path(p) for p in json.loads((P/(stem+'-manifest-paths.json')).read_text())]
for p in paths:p.write_text(json.dumps(fix(json.loads(p.read_text())),ensure_ascii=False,indent=1)+'\n')
p=P/'near_pass76_prepare.py';p.write_text(p.read_text().replace(old,new))
report=dict(reason='Removed erroneous Marathi label from the Marwari kuto comparison. The parent and relation are unchanged.',affectedIds=sorted(ids),overlayShaBefore=sha(raw),overlayShaAfter=sha(payload),ancestryUnchanged=True)
(P/(stem+'-note-correction.json')).write_text(json.dumps(report,indent=2)+'\n');p=P/(stem+'-validation.json');d=json.loads(p.read_text());d['postSaveEvidenceCorrection']=report;p.write_text(json.dumps(d,indent=2)+'\n')
