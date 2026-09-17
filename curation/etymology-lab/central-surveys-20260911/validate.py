"""Validate research proposals on a temporary graph; never mutate accepted data."""
import csv,json,sys,hashlib,tempfile,shutil,datetime
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];sys.path.insert(0,str(root))
from assign_form_ids import validate_assignments,apply_assignments
watched=[root/'cldf/forms.csv',root/'cldf/edges.csv',root/'data/etymology-assignments.csv']
def stamps():
 return {str(f):(f.stat().st_size,f.stat().st_mtime_ns) for f in watched}
before=stamps()
props=[]
for lid in ['mewari_basad','Nimadi','bagheli_lakshman']:
 for f in (p.parent/lid).glob('batch-*.json'):
  x=json.loads(f.read_text())
  if x.get('researchDirectory')==str(p):props+=x['proposals']
rows=[r for x in props for r in x['assignments']];targets={r['Form_ID'] for r in rows};parents={r['Etymon_ID'] for r in rows}
from collections import defaultdict
bytarget=defaultdict(list)
for r in rows:bytarget[r['Form_ID']].append(r)
for fid, rr in bytarget.items():
 if len(rr)>1:
  assert all(r['Kind']=='component' for r in rr),fid
  assert sorted(int(r['Pos']) for r in rr)==list(range(1,len(rr)+1)),fid
  assert len({r['Etymon_ID'] for r in rr})==len(rr),fid
forms=list(csv.DictReader(open(root/'cldf/forms.csv')));byid={r['ID']:r for r in forms}
missing=sorted((targets|parents)-byid.keys());report={'proposals':len(props),'rows':len(rows),'records':len(targets),'missing_current_ids':missing}
report['checked_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
changed=[]
for x in props:
 for r in x['records']:
  current=byid.get(r['ID'])
  if current:
   fields={k:{'frozen':r.get(k),'current':current.get(k)} for k in ['Language_ID','Form','Gloss'] if r.get(k)!=current.get(k)}
   if fields:changed.append({'id':r['ID'],'fields':fields})
report['changed_target_records']=changed
registry={r['Form_ID'] for r in csv.DictReader(open(root/'data/form-identities.csv'))};report['missing_registry_targets']=sorted(targets-registry)
overlay=list(csv.DictReader(open(root/'data/etymology-assignments.csv')))
conflicts=[r for r in overlay if r['Form_ID'] in targets and r['Rank']=='1' and r['Status']=='accepted'];report['current_overlay_target_rows']=conflicts
assert not conflicts,'Concurrent accepted rows need reconciliation'
if missing or changed:
 report['graph_validation']='deferred: missing or changed records require alias/redirect or concurrent-build reconciliation before acceptance'
else:
 blocked=[r for r in rows if byid[r['Etymon_ID']].get('Status')=='unlinked' and r['Etymon_ID'] not in targets]
 report['unlinked_parent_dependencies']=sorted({r['Etymon_ID'] for r in blocked})
 report['dependency_blocked_rows']=len(blocked)
 eligible=[r for r in rows if r not in blocked]
 validate_assignments(forms,eligible)
 with tempfile.TemporaryDirectory(prefix='central-surveys-validation-') as d:
  graph=Path(d)/'edges.csv';shutil.copyfile(root/'cldf/edges.csv',graph)
  first=apply_assignments(graph,forms,eligible);second=apply_assignments(graph,forms,eligible)
  assert second==0,second
  edges=list(csv.DictReader(open(graph)));actual={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank'],r.get('Pos','')) for r in edges}
  assert all((r['Form_ID'],r['Etymon_ID'],r['Kind'],r['Rank'],r.get('Pos','')) in actual for r in eligible)
  report.update(graph_validation='passed eligible subset; donor dependencies remain' if blocked else 'passed on temporary graph',validated_rows=len(eligible),first_application_changes=first,second_application_changes=second)
report['overlay_sha256']=hashlib.sha256((root/'data/etymology-assignments.csv').read_bytes()).hexdigest()
report['input_files_unchanged_during_check']=before==stamps()
if not report['input_files_unchanged_during_check']:
 report['graph_validation']='deferred: corpus files changed during validation; rerun on stable inputs'
(p/'validation.json').write_text(json.dumps(report,indent=2))
if report['graph_validation']=='passed on temporary graph':
 (p/'validation-last-passed.json').write_text(json.dumps(report,indent=2))
summary={k:(len(v) if isinstance(v,list) else v) for k,v in report.items()}
print(json.dumps(summary,indent=2))
