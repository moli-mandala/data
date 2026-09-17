import csv,json,hashlib,io,os,tempfile,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;root=P.parents[2];overlay=root/'data/etymology-assignments.csv'
a='including Tirahi-area and Pashai/Torwali comparanda';b='including Pashai and Torwali comparanda'
raw=overlay.read_bytes();reader=csv.DictReader(io.StringIO(raw.decode()));fields=reader.fieldnames;old=list(reader)
rows=[dict(r) for r in old];changed=[]
for r in rows:
 if a in r['Notes']:
  assert r['Etymon_ID']=='5110';r['Notes']=r['Notes'].replace(a,b);changed.append(r['Form_ID'])
assert len(changed)==5
shutil.copyfile(overlay,P/'backups/etymology-assignments-before-seventeenth-note-correction.csv')
paths=[P/('global-seventeenth-'+x+'.json') for x in ['rules','decisions','saved-assignments']]+[Path(x) for x in json.loads((P/'global-seventeenth-manifest-paths.json').read_text())]
for p in paths:
 s=p.read_text()
 if a in s:p.write_text(s.replace(a,b))
p=P/'global_seventeenth_prepare.py';p.write_text(p.read_text().replace(a,b))
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(rows)
with tempfile.NamedTemporaryFile(dir=overlay.parent,delete=False) as f:f.write(s.getvalue().encode());temp=Path(f.name)
assert overlay.read_bytes()==raw;os.replace(temp,overlay)
assert list(csv.DictReader(overlay.open()))==rows
assert all(r==s or (r['Form_ID'] in changed and all(r[k]==s[k] for k in fields if k!='Notes')) for r,s in zip(old,rows))
report=dict(reason='Remove an unsupported geographic qualifier from CDIAL 5110 evidence; primary entry names Pashai and Torwali, not Tirahi.',affectedRecords=changed,relationshipsUnchanged=True,allOtherRowsAndColumnsPreserved=True,overlayHashBefore=hashlib.sha256(raw).hexdigest(),overlayHashAfter=hashlib.sha256(overlay.read_bytes()).hexdigest())
(P/'global-seventeenth-note-correction.json').write_text(json.dumps(report,indent=2)+'\n')
p=P/'global-seventeenth-validation.json';v=json.loads(p.read_text());v['postSaveEvidenceCorrection']=report;p.write_text(json.dumps(v,indent=2)+'\n')
print(report)
