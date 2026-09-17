import csv,json,io,os,hashlib,shutil
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');root=P.parents[2];f=root/'data/etymology-assignments.csv'
oldtext=f.read_bytes();rows=list(csv.DictReader(io.StringIO(oldtext.decode())));fields=list(rows[0]);saved=json.loads((P/'pass229-saved-assignments.json').read_text());target={r['Form_ID'] for r in saved}
changed=[]
for r in rows:
 if r['Form_ID'] in target and 'Bhatri phūth' in r['Notes']:
  old=r.copy();r['Notes']=r['Notes'].replace('Bhatri phūth','Bhateri phūth');changed.append(dict(before=old,after=r.copy()))
assert len(changed)==10
backup=P/'backups/etymology-assignments-before-pass229-note-correction.csv';assert not backup.exists() or backup.read_bytes()==oldtext;backup.write_bytes(oldtext)
paths=[P/'pass229-decisions.json',P/'pass229-rules.json',P/'pass229-saved-assignments.json',P/'pass229_prepare.py']
for item in json.loads((P/'pass229-manifest-paths.json').read_text()):
 x=Path(item);paths.extend([x,x.with_name(x.stem+'-review.md')])
for x in paths:
 s=x.read_text() if x.exists() else '';
 if not x.exists():continue
 x.write_text(s.replace('Bhatri phūth','Bhateri phūth'))
assert f.read_bytes()==oldtext
out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)
tmp=f.with_suffix('.note-correction.tmp');tmp.write_text(out.getvalue(),newline='');os.replace(tmp,f)
current=list(csv.DictReader(f.open()));assert current==rows
original=list(csv.DictReader(io.StringIO(oldtext.decode())))
for a,b in zip(original,current):assert {k:v for k,v in a.items() if k!='Notes'}=={k:v for k,v in b.items() if k!='Notes'}
assert all(a==b for a,b in zip(original,current) if a['Form_ID'] not in target)
(P/'pass229-note-correction.json').write_text(json.dumps(dict(reason='Correct language label bhatr to Bhateri, as recorded in cldf/languages.csv. No lexical assignment or source record changed.',changedRows=changed,beforeHash=hashlib.sha256(oldtext).hexdigest(),afterHash=hashlib.sha256(f.read_bytes()).hexdigest(),nonNoteFieldsUnchanged=True,unrelatedRowsUnchanged=True,originalSaveValidationPreserved=True),ensure_ascii=False,indent=1)+'\n')
print('Corrected 10 shared evidence notes; all non-note fields and unrelated rows unchanged.')
