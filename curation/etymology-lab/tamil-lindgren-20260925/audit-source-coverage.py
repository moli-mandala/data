"""Reconcile Tamil Lindgren source IDs with pending decisions and manifests."""
import csv,json
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text())
ids=[r['ID'] for r in records];assert len(ids)==244 and len(set(ids))==244
with (R/'cldf/forms.csv').open() as f:
 source=[r for r in csv.DictReader(f) if r['Language_ID']=='Tamil' and r['Source'].startswith('lindgren2023dravidian')]
assert len(source)==244 and set(ids)=={r['ID'] for r in source}
d=json.loads((P/'proposed-decisions.json').read_text())
owner=Counter(x['record']['ID'] for x in d['accepted']+d['held'])
assert set(owner)==set(ids) and all(v==1 for v in owner.values())
paths=json.loads((P/'review-manifest-paths.json').read_text())
man=Counter(r['ID'] for path in paths for prop in json.loads(Path(path).read_text())['proposals'] for r in prop['records'])
assert man==Counter(x['record']['ID'] for x in d['accepted'])
rows=sum(len(prop['assignments']) for path in paths for prop in json.loads(Path(path).read_text())['proposals'])
out={'compiledSourceRecords':244,'stagedRecords':len(records),'proposedRecords':len(d['accepted']),'heldRecords':len(d['held']),'unexaminedRecords':0,'pendingBatches':len(paths),'assignmentRows':rows,'exactOwnership':True,'manifestAgreement':True}
(P/'source-coverage-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
