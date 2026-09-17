import csv,json,collections,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
read=lambda f:json.loads((P/f).read_text())
prefixes=[f.removesuffix('manifest-paths.json') for f in read('pass-ledger.json')['manifestFiles']]
rows=[r for pre in prefixes for r in read(pre+'saved-assignments.json')]
manifestrows=[]
for pre in prefixes:
 for path in read(pre+'manifest-paths.json'):
  f=Path(path);assert f.with_name(f.stem+'-review.md').exists()
  for q in json.loads(f.read_text())['proposals']:manifestrows.extend(q['assignments'])
key=lambda r:tuple(sorted(r.items()))
assert collections.Counter(map(key,rows))==collections.Counter(map(key,manifestrows))
old=list(csv.DictReader((P/'backups/etymology-assignments-before.csv').open()))
current=list(csv.DictReader((ROOT/'data/etymology-assignments.csv').open()))
assert current==old+rows
status=read('status.json');assert status['alreadyLinked']+status['savedAffectedRecords']+status['heldRecords']+status['unresearchedRecords']==status['scopeRecords']
assert status['savedAssignments']+status['additionalDictionaryAssignments']==len(rows)
excluded=set(read('scope-correction.json')['excludedIds'])
assert status['savedAffectedRecords']==len({r['Form_ID'] for r in rows if r['Form_ID'] not in excluded})
reports=[read(pre+'validation.json') for pre in prefixes];assert all(x['secondApplicationChanges']==0 for x in reports)
report=dict(totalNewRows=len(rows),allPriorOverlayRowsPreserved=len(old),exactOverlayAndManifestMatch=True,scopePartitionComplete=True,firstApplicationChanges=sum(x['firstApplicationChanges'] for x in reports),allSecondApplicationsZero=True,overlaySha256=hashlib.sha256((ROOT/'data/etymology-assignments.csv').read_bytes()).hexdigest(),status=status)
(P/'latest-joint-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
