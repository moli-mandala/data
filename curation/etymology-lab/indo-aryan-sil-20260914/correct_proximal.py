"""Withdraw only four own unsupported proximal assignments after full primary review."""
import csv,json,sys,tempfile,shutil,hashlib,datetime,os
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT))
from assign_form_ids import validate_assignments,apply_assignments
ids={'f_b24vhtsq6trag','f_cirlwby2um4z4','f_mdkwky6gitydq','f_i5ue2qvkhymao'}
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
overlay=ROOT/'data/etymology-assignments.csv';before=sha(overlay);watch=[ROOT/'cldf/forms.csv',ROOT/'cldf/edges.csv',ROOT/'data/form-identities.csv'];hashes={str(p):sha(p) for p in watch}
with overlay.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;old=list(r)
saved=read(P/'global-second-saved-assignments.json');removed=[r for r in saved if r['Form_ID'] in ids];assert len(removed)==4;assert [r for r in old if r['Form_ID'] in ids]==removed;retained=[r for r in saved if r['Form_ID'] not in ids];new=[r for r in old if r['Form_ID'] not in ids]
forms=[{'ID':r['ID'],'Status':r['Status']} for r in csv.DictReader((ROOT/'cldf/forms.csv').open())];validate_assignments(forms,retained)
with tempfile.TemporaryDirectory() as td:
 f=Path(td)/'edges.csv';shutil.copyfile(ROOT/'cldf/edges.csv',f);first=apply_assignments(f,forms,retained);second=apply_assignments(f,forms,retained);assert second==0
 actual=[r for r in csv.DictReader(f.open()) if r['Child_ID'] in {x['Form_ID'] for x in retained} and r['Rank']=='1'];assert len(actual)==len(retained)
 assert {(r['Child_ID'],r['Parent_ID'],r['Kind']) for r in actual}=={(r['Form_ID'],r['Etymon_ID'],r['Kind']) for r in retained}
backup=P/'backups/proximal-correction';backup.mkdir(exist_ok=False);shutil.copyfile(overlay,backup/'overlay-before.csv')
for name in ['global-second-decisions.json','global-second-saved-assignments.json','global-second-validation.json']:shutil.copyfile(P/name,backup/name)
with tempfile.NamedTemporaryFile(mode='w',dir=overlay.parent,delete=False,newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(new);f.flush();os.fsync(f.fileno());temp=Path(f.name)
assert sha(overlay)==before;os.replace(temp,overlay)
with overlay.open(newline='') as f:assert list(csv.DictReader(f))==new
reason='Withdrawn after fuller primary review: Chatterji (1926), Part II, pp. 832–833, distinguishes Rajasthani ā in the eta paradigm from Gujarati ā derived from ayam through aya/āa. The Bhil/Kaithal bare ā survey response does not decide those competing stems; do not treat this as merely cross-IA donor uncertainty.'
d=read(P/'global-second-decisions.json')
for x in d['accepted']:
 if x['record']['ID'] in ids:d['held'].append(dict(record=x['record'],families=[x['family']],reason=reason,passNumber=19,withdrawnAssignment=x))
d['accepted']=[x for x in d['accepted'] if x['record']['ID'] not in ids];write(P/'global-second-decisions.json',d);write(P/'global-second-saved-assignments.json',retained)
for path in read(P/'global-second-manifest-paths.json'):
 f=Path(path);m=read(f)
 if not any(ids & set(q['formIds']) for q in m['proposals']):continue
 shutil.copyfile(f,backup/(m['language']+'-'+f.name))
 for q in m['proposals']:
  q['formIds']=[i for i in q['formIds'] if i not in ids];q['records']=[r for r in q['records'] if r['ID'] not in ids];q['assignments']=[r for r in q['assignments'] if r['Form_ID'] not in ids]
 m['proposals']=[q for q in m['proposals'] if q['formIds']];m['editorialCorrection']=reason;write(f,m)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();report=read(P/'global-second-validation.json');report.update(assignmentRows=len(retained),affectedRecords=len(retained),firstApplicationChanges=first,secondApplicationChanges=second,revalidatedAt=now,editorialCorrection=reason,withdrawnRecords=sorted(ids),originalValidation=str(backup/'global-second-validation.json'),overlayHashAfter=sha(overlay));write(P/'global-second-validation.json',report)
assert all(sha(Path(p))==v for p,v in hashes.items())
write(P/'proximal-correction.json',dict(at=now,reason=reason,withdrawnRows=removed,overlayBefore=before,overlayAfter=sha(overlay),unrelatedRowsPreserved=True,retainedRowsRevalidated=len(retained),firstApplicationChanges=first,secondApplicationChanges=second,sharedFileHashes=hashes))
text=Path('/tmp/chatterji-odbl.txt').read_text();write(P/'chatterji-pronoun-primary.json',dict(author='Suniti Kumar Chatterji',title='The Origin and Development of the Bengali Language, Part II',year=1926,url='https://ir.nbu.ac.in/server/api/core/bitstreams/c2a2a9b5-8a1a-4068-af64-97039256b8f9/content',pages=[dict(pdfPage=n,printedPage=n+648,text=text.split('PDF PAGE '+str(n)+'\n')[1].split('PDF PAGE '+str(n+1)+'\n')[0]) for n in range(182,187)],visualVerification=[184,185]))
print('Withdrawn',len(removed),'retained',len(retained),'repeat',second)
