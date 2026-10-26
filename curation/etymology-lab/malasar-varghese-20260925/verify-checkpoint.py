"""Independent Malasar exact-source and pending-manifest validation."""
import collections,csv,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;DATA=P.parents[2]
sys.path.insert(0,str(DATA))
from etymology_lab import rows_from_decisions
from etymology_assignments import read_assignments
log=(P/'check-pass-001.log').read_text();v=json.JSONDecoder().raw_decode(log[log.index('{'):])[0]
assert v['dryRun'] and v['secondApplicationChanges']==0
new=json.loads((P/'proposed-decisions.json').read_text());expected=rows_from_decisions(new['accepted'],'')
paths=json.loads((P/'review-manifest-paths.json').read_text());actual=[]
for path in paths:
 m=json.loads(Path(path).read_text());assert m['status']=='pending-review' and m['researchDirectory']==str(P)
 for proposal in m['proposals']:
  assert proposal['status']=='pending-review';actual+=proposal['assignments']
key=lambda r:tuple(str(r.get(k,'') or '') for k in ('Form_ID','Etymon_ID','Kind','Rank','Pos'))
assert collections.Counter(map(key,actual))==collections.Counter(map(key,expected))
assert len(actual)==v['assignmentRows']==66 and len(new['accepted'])==v['affectedRecords']==63
source={r['ID']:r for r in csv.DictReader(open(DATA/'cldf/forms.csv')) if r['Language_ID']=='Malasar' and 'varghese2015palakkad' in r['Source'] and r['Status'] not in ('entry','redirect')}
linked={r['Child_ID'] for r in csv.DictReader(open(DATA/'cldf/edges.csv')) if r['Rank']=='1'}
linked.update(r['Form_ID'] for r in read_assignments() if str(r.get('Rank'))=='1')
linked=set(source)&linked
old=json.loads((P.parent/'dravidian-surveys-20260921/proposed-decisions.json').read_text())
prior_a={x['record']['ID'] for x in old['accepted'] if x['record']['Language_ID']=='Malasar' and 'varghese2015palakkad' in x['record']['Source']}
prior_h={x['record']['ID'] for x in old['held'] if x['record']['Language_ID']=='Malasar' and 'varghese2015palakkad' in x['record']['Source']}
new_a={x['record']['ID'] for x in new['accepted']};new_h={x['record']['ID'] for x in new['held']};sets=[linked,prior_a,prior_h,new_a,new_h]
assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]);assert set(source)==set.union(*sets)
assert [len(x) for x in sets]==[0,285,17,63,81]
report=dict(source='varghese2015palakkad',language='Malasar',sourceRecords=len(source),priorLinked=len(linked),priorProposed=len(prior_a),priorHeld=len(prior_h),proposedRecords=len(new_a),heldRecords=len(new_h),unexaminedRecords=0,uniqueOwnership=True,manifestAgreement=True,assignmentRows=len(actual),validation=v)
(P/'checkpoint-001-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({k:report[k] for k in ('sourceRecords','priorLinked','priorProposed','priorHeld','proposedRecords','heldRecords','assignmentRows','manifestAgreement')})
