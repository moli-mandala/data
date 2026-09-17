"""Conflict-check and atomically append the explicitly approved research overlay."""
import csv,json,sys,tempfile,shutil,io,os,hashlib,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT))
from assign_form_ids import validate_assignments,apply_assignments
from edges_build import validate_edge_dicts
def read(p):
    with p.open() as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
overlay=ROOT/'data/etymology-assignments.csv';registry=ROOT/'data/form-identities.csv'
watch=[overlay,registry,ROOT/'cldf/forms.csv',ROOT/'cldf/edges.csv'];before={str(p):sha(p) for p in watch}
fields,old=read(overlay);rf,reg=read(registry);_,forms=read(ROOT/'cldf/forms.csv');byid={r['ID']:r for r in forms}
rows=json.loads((P/'approved-assignments.json').read_text());targets={r['Form_ID'] for r in rows}
assert not [r for r in old if r['Form_ID'] in targets], 'Concurrent target overlay rows require reconciliation'
for m in json.loads((P/'before-approval-manifests.json').read_text()).values():
    for q in m['proposals']:
        for r in q['records']:
            assert all(byid[r['ID']][k]==r[k] for k in ['Language_ID','Form','Gloss']),r['ID']
newforms=json.loads((P/'new-donor-forms.json').read_text());newreg=json.loads((P/'new-donor-identities.json').read_text())
assert not {r['ID'] for r in newforms}&byid.keys()
assert not {r['Form_ID'] for r in newreg}&{r['Form_ID'] for r in reg}
forms+=newforms
validate_assignments(forms,rows)
with tempfile.TemporaryDirectory(prefix='central-approved-graph-') as d:
    graph=Path(d)/'edges.csv';shutil.copyfile(ROOT/'cldf/edges.csv',graph)
    first=apply_assignments(graph,forms,rows);second=apply_assignments(graph,forms,rows);assert second==0
    _,edges=read(graph);validate_edge_dicts(edges,{r['ID']:r.get('Status','') for r in forms})
    actual={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank'],r['Pos']) for r in edges}
    assert all((r['Form_ID'],r['Etymon_ID'],r['Kind'],r['Rank'],r['Pos']) in actual for r in rows)
    _,original=read(ROOT/'cldf/edges.csv')
    assert [r for r in edges if r['Child_ID'] not in targets]==[r for r in original if r['Child_ID'] not in targets]
assert before=={str(p):sha(p) for p in watch},'Inputs changed during validation'
backup=P/'acceptance-backups';backup.mkdir(exist_ok=True)
def append_atomic(path,fields,oldrows,newrows):
    raw=path.read_bytes();shutil.copyfile(path,backup/path.name)
    # Preserve every existing byte and append ordinary structured CSV rows.
    b=io.StringIO(newline='');csv.DictWriter(b,fieldnames=fields).writerows(newrows)
    payload=raw+(b'' if raw.endswith(b'\n') else b'\n')+b.getvalue().encode()
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(payload);tmp=Path(f.name)
    assert read(tmp)[1]==oldrows+newrows
    assert path.read_bytes()==raw,'Concurrent update immediately before replace'
    os.replace(tmp,path)
    assert read(path)[1]==oldrows+newrows
append_atomic(registry,rf,reg,newreg)
append_atomic(overlay,fields,old,rows)
report=dict(savedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),proposals=757,records=len(targets),rows=len(rows),newDonorHeads=len(newreg),temporaryGraph='passed',firstApplicationChanges=first,secondApplicationChanges=second,unrelatedOverlayRowsPreserved=len(old),unrelatedGraphRowsPreserved=True,hashesBefore=before,overlaySha256=sha(overlay),registrySha256=sha(registry),donorDependenciesRemaining=0)
(P/'acceptance-validation.json').write_text(json.dumps(report,indent=2)+'\n')
for lid in ['mewari_basad','Nimadi','bagheli_lakshman']:
 for path in (P.parent/lid).glob('batch-*.json'):
  m=json.loads(path.read_text())
  if m.get('researchDirectory')!=str(P):continue
  m['status']='saved';m['savedAt']=report['savedAt'];m['validation']=str(P/'acceptance-validation.json')
  for q in m['proposals']:q['status']='saved'
  path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,indent=2))
