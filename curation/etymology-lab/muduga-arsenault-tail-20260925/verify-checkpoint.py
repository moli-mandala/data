"""Independently verify Muduga tail source ownership and manifest vs dry run."""
import collections,csv,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;DATA=P.parents[2]
sys.path.insert(0,str(DATA))
from etymology_lab import rows_from_decisions
from etymology_assignments import read_assignments
log=(P/'check-pass-001.log').read_text();v=json.JSONDecoder().raw_decode(log[log.index('{'):])[0]
assert v['dryRun'] and v['assignmentRows']==2 and v['affectedRecords']==2 and v['secondApplicationChanges']==0
d=json.loads((P/'proposed-decisions.json').read_text());expected=rows_from_decisions(d['accepted'],'')
m=json.loads((P.parent/'Muduga/batch-029.json').read_text());assert m['status']=='pending-review' and m['researchDirectory']==str(P)
actual=[row for x in m['proposals'] for row in x['assignments']]
key=lambda x:tuple(str(x.get(k,'') or '') for k in ['Form_ID','Etymon_ID','Kind','Rank','Pos'])
assert collections.Counter(map(key,actual))==collections.Counter(map(key,expected))
all_source={r['ID']:r for r in csv.DictReader(open(DATA/'cldf/forms.csv')) if 'arsenault-abraham2022muduga' in r['Source'] and r['Language_ID']=='Muduga' and r['Status'] not in ('entry','redirect')}
linked={r['Child_ID'] for r in csv.DictReader(open(DATA/'cldf/edges.csv')) if r['Rank']=='1'}
linked.update(r['Form_ID'] for r in read_assignments() if str(r.get('Rank'))=='1')
prior=set(all_source)&linked
new={x['record']['ID'] for x in d['accepted']};held={x['record']['ID'] for x in d['held']}
assert len(all_source)==109 and len(prior)==99 and len(new)==2 and len(held)==8
assert not(prior&new or prior&held or new&held)
assert set(all_source)==prior|new|held
report=dict(source='arsenault-abraham2022muduga',language='Muduga',sourceRecords=len(all_source),priorLinked=len(prior),proposedRecords=len(new),heldRecords=len(held),unexaminedRecords=0,uniqueOwnership=True,manifestAgreement=True,assignmentRows=len(actual),validation=v)
(P/'checkpoint-001-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(P/'source-coverage-001.json').write_text(json.dumps({k:report[k] for k in ['source','language','sourceRecords','priorLinked','proposedRecords','heldRecords','unexaminedRecords','uniqueOwnership']},ensure_ascii=False,indent=2)+'\n')
(P/'review-manifest-paths.json').write_text(json.dumps([str(P.parent/'Muduga/batch-029.json')],indent=2)+'\n')
print({k:report[k] for k in ['sourceRecords','priorLinked','proposedRecords','heldRecords','assignmentRows','manifestAgreement']})
