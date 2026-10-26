"""Independently reconcile full Tulu Lindgren source and bounded review checkpoint."""
import collections,csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());ids=[r['ID'] for r in records]
assert len(ids)==1730 and len(set(ids))==1730
with (R/'cldf/forms.csv').open() as fh:
    compiled=[r for r in csv.DictReader(fh) if r['Language_ID']=='Tulu' and r['Source'].startswith('lindgren2023dravidian[')]
assert len(compiled)==1730 and set(ids)=={r['ID'] for r in compiled}
d=json.loads((P/'proposed-decisions.json').read_text());u=json.loads((P/'unexamined-inventory.json').read_text())
all_rows=d['accepted']+d['held']+u
owners=collections.Counter(x['record']['ID'] for x in all_rows)
assert set(owners)==set(ids) and all(v==1 for v in owners.values())
paths=json.loads((P/'review-manifest-paths.json').read_text())
man=collections.Counter(r['ID'] for path in paths for prop in json.loads(Path(path).read_text())['proposals'] for r in prop['records'])
assert man==collections.Counter(x['record']['ID'] for x in d['accepted'])
rows=sum(len(prop['assignments']) for path in paths for prop in json.loads(Path(path).read_text())['proposals'])
assert rows==len(d['accepted'])
coverage=json.loads((P/'source-coverage-001.json').read_text())
assert coverage['recordOwnership']=={x['record']['ID']:x['status'].replace('pending-review','proposed').replace('research-hold','held') for x in all_rows}
out={'compiledSourceRecords':1730,'stagedRecords':len(records),'proposedRecords':len(d['accepted']),'heldRecords':len(d['held']),'unexaminedRecords':len(u),'pendingBatches':len(paths),'assignmentRows':rows,'exactOwnership':True,'manifestAgreement':True}
(P/'source-coverage-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
