"""Validate and atomically save the user's authorized joint survey pass.
Preserves every unrelated overlay row, compiled file, and identity-registry byte.
"""
import csv,json,sys,io,os,tempfile,shutil,hashlib,datetime,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT))
from assign_form_ids import validate_assignments,apply_assignments

def read(p):
 with p.open(newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
d=json.loads((P/'pass261-decisions.json').read_text());acc=d['accepted'];targets={x['record']['ID'] for x in acc};assert len(targets)==len(acc)
overlay=ROOT/'data/etymology-assignments.csv';registry=ROOT/'data/form-identities.csv';watch=[overlay,registry,ROOT/'cldf/forms.csv',ROOT/'cldf/edges.csv'];before={str(x):sha(x) for x in watch}
fields,old=read(overlay);assert not [r for r in old if r['Form_ID'] in targets], 'Concurrent/alternate target assignments require explicit reconciliation'
needed=targets|{x['parent'] for x in acc};forms=[];selected={}
for r in csv.DictReader((ROOT/'cldf/forms.csv').open()):
 forms.append({'ID':r['ID'],'Status':r['Status']})
 if r['ID'] in needed:selected[r['ID']]=r
for x in acc:
 r=selected[x['record']['ID']]
 assert not r['Redirect']
 assert all(r[k]==x['record'][k] for k in ['Language_ID','Form','Gloss','Source','Tags']),r['ID']
 assert not selected[x['parent']]['Redirect'],x['parent']
 assert selected[x['parent']]['Status']!='unlinked',x['parent'] # Verified existing lexical donors need not already have their own etymology.
rows=[dict(Form_ID=x['record']['ID'],Etymon_ID=x['parent'],Kind=x.get('kind','reflex'),Rank='1',Status='accepted',Source=x['citation'],Notes=x['evidence']+' Joint SIL review 2026-09-14.',Pos='') for x in acc]
validate_assignments(forms,rows)
# Confirm every target has a registered durable identity without loading the registry.
registered=set()
for r in csv.DictReader(registry.open()):
 if r['Form_ID'] in targets:registered.add(r['Form_ID'])
assert registered==targets,targets-registered
with tempfile.TemporaryDirectory(prefix='sil-joint-graph-') as td:
 graph=Path(td)/'edges.csv';shutil.copyfile(ROOT/'cldf/edges.csv',graph)
 first=apply_assignments(graph,forms,rows);second=apply_assignments(graph,forms,rows);assert second==0
 # Compare every unrelated edge as parsed CSV. No shared compiled graph is written.
 _,original=read(ROOT/'cldf/edges.csv');_,result=read(graph)
 assert not [e for e in original if e['Child_ID'] in targets and e['Rank']=='1']
 assert [e for e in result if e['Child_ID'] not in targets]==[e for e in original if e['Child_ID'] not in targets]
 expected={(r['Form_ID'],r['Etymon_ID'],r['Kind'],r['Rank'],r['Pos']) for r in rows}
 actual={(e['Child_ID'],e['Parent_ID'],e['Kind'],e['Rank'],e['Pos']) for e in result if e['Child_ID'] in targets and e['Rank']=='1'}
 assert actual==expected
 assert all(r['Status']=='' for r in forms if r['ID'] in targets)
 del original,result
assert before=={str(x):sha(x) for x in watch},'Inputs changed during validation; retry from fresh input'
backup=P/'backups';backup.mkdir(exist_ok=True);shutil.copyfile(overlay,backup/'etymology-assignments-before-pass261.csv')
# Read again immediately before replace; refuse all changes since validation.
fields2,old2=read(overlay);assert fields2==fields and old2==old and sha(overlay)==before[str(overlay)]
raw=overlay.read_bytes();out=io.StringIO(newline='');csv.DictWriter(out,fieldnames=fields).writerows(rows)
payload=raw+(b'' if raw.endswith(b'\n') else b'\n')+out.getvalue().encode()
with tempfile.NamedTemporaryFile(dir=overlay.parent,delete=False) as f:
 f.write(payload);f.flush();os.fsync(f.fileno());tmp=Path(f.name)
assert read(tmp)[1]==old+rows
assert sha(overlay)==before[str(overlay)]
os.replace(tmp,overlay)
assert read(overlay)[1]==old+rows
assert all(sha(x)==before[str(x)] for x in watch if x!=overlay)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=dict(savedAt=now,assignmentRows=len(rows),affectedRecords=len(targets),parentNodes=len({r['Etymon_ID'] for r in rows}),previousOverlayRows=len(old),firstApplicationChanges=first,secondApplicationChanges=second,unrelatedEdgesPreserved=True,sourceFormsPreserved=True,identityRegistryPreserved=True,hashesBefore=before,overlayHashAfter=sha(overlay),checks=['validate_assignments','persistent registry membership','exact target content/source/tags','all unrelated graph edges identical','exact intended rank-1 edges','repeat application changes zero','all original overlay rows identical','compiled forms/edges and registry hashes unchanged'])
(P/'pass261-saved-assignments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1))
(P/'pass261-validation.json').write_text(json.dumps(report,indent=2)+'\n')
# Required language-scoped manifests, grouped by exact parent and analysis.
bylang=collections.defaultdict(list)
for x in acc:bylang[x['record']['Language_ID']].append(x)
manifest_paths=[]
for lid,xs in sorted(bylang.items()):
 dest=P.parent/lid;dest.mkdir(exist_ok=True)
 nums=[int(f.stem.split('-')[1]) for f in dest.glob('batch-*.json') if f.stem.split('-')[1].isdigit()];n=max(nums,default=0)+1
 grouped=collections.defaultdict(list)
 for x in xs:grouped[(x['parent'],x['citation'],x['evidence'],x.get('kind','reflex'))].append(x)
 proposals=[]
 for j,((parent,cite,ev,kind),ys) in enumerate(grouped.items(),1):
  ids={y['record']['ID'] for y in ys}
  proposals.append(dict(number=j,status='saved',parentId=parent,parentForm=selected[parent]['Form'],kind=kind,citation=cite,evidence=ev,formIds=sorted(ids),records=[y['record'] for y in ys],assignments=[r for r in rows if r['Form_ID'] in ids]))
 path=dest/f'batch-{n:03d}.json';assert not path.exists()
 path.write_text(json.dumps(dict(language=lid,batch=n,status='saved',savedAt=now,authorization='User: etymologise all Indo-Aryan SIL surveys jointly; do everything; surface tough/unclear cases for audit once done.',researchDirectory=str(P),validation=str(P/'pass261-validation.json'),proposals=proposals),ensure_ascii=False,indent=1)+'\n');manifest_paths.append(str(path))
(P/'pass261-manifest-paths.json').write_text(json.dumps(manifest_paths,indent=2)+'\n')
print(json.dumps(report,indent=2))
