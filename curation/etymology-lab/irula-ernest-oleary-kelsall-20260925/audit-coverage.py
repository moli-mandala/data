"""Independently reconcile this checkpoint with all 2,054 source identities."""
import collections,csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
base=json.loads((P/'ownership-baseline.json').read_text())
records=json.loads((P/'survey-inventory.json').read_text());ids=[x['ID'] for x in records]
assert len(ids)==len(set(ids))==822
with (R/'cldf/forms.csv').open() as fh:
    compiled=[r for r in csv.DictReader(fh) if r['Language_ID']=='Irula' and r['Source'].startswith('ernest-oleary-kelsall2018irula[')]
source_ids={x['ID'] for x in compiled};assert len(source_ids)==2054
prior_p=set(base['priorPendingIds']);prior_h=set(base['priorHeldIds']);assert len(prior_p)==1139 and len(prior_h)==93
assert not prior_p&prior_h and not (prior_p|prior_h)&set(ids)
assert prior_p|prior_h|set(ids)==source_ids
d=json.loads((P/'proposed-decisions.json').read_text());u=json.loads((P/'unexamined-inventory.json').read_text())
own=collections.Counter(x['record']['ID'] for x in d['accepted']+d['held']+u)
assert set(own)==set(ids) and all(v==1 for v in own.values())
paths=json.loads((P/'review-manifest-paths.json').read_text())
manifest=collections.Counter(r['ID'] for path in paths for p in json.loads(Path(path).read_text())['proposals'] for r in p['records'])
assert manifest==collections.Counter(x['record']['ID'] for x in d['accepted'])
rows=sum(len(p['assignments']) for path in paths for p in json.loads(Path(path).read_text())['proposals'])
assert rows==len(d['accepted'])
c=json.loads((P/'source-coverage-001.json').read_text())
assert c['recordOwnership']=={x['record']['ID']:x['status'].replace('pending-review','proposed').replace('research-hold','held') for x in d['accepted']+d['held']+u}
report={'sourceRecords':2054,'priorProposed':1139,'priorHeld':93,'residualRecords':822,'proposedRecords':len(d['accepted']),'heldRecords':len(d['held']),'unexaminedRecords':len(u),'pendingBatches':len(paths),'assignmentRows':rows,'exactOwnership':True,'manifestAgreement':True}
(P/'source-coverage-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
