"""Independent whole-source, pending-manifest and canonical dry-run audit."""
import collections
import csv
import json
import sys
from pathlib import Path

p = Path(__file__).resolve().parent
data = p.parents[2]
sys.path.insert(0, str(data))
from etymology_lab import rows_from_decisions
from etymology_assignments import read_assignments

log = (p / 'check-pass-001.log').read_text()
validation = json.JSONDecoder().raw_decode(log[log.index('{'):])[0]
assert validation['dryRun'] and validation['secondApplicationChanges'] == 0
new = json.loads((p / 'proposed-decisions.json').read_text())
expected = rows_from_decisions(new['accepted'], '')
paths = json.loads((p / 'review-manifest-paths.json').read_text())
actual = []
for path in paths:
    manifest = json.loads(Path(path).read_text())
    assert manifest['language'] == 'Maliyad'
    assert manifest['status'] == 'pending-review' and manifest['researchDirectory'] == str(p)
    for proposal in manifest['proposals']:
        assert proposal['status'] == 'pending-review'
        actual.extend(proposal['assignments'])
key = lambda row: tuple(str(row.get(k, '') or '') for k in ('Form_ID','Etymon_ID','Kind','Rank','Pos'))
assert collections.Counter(map(key, actual)) == collections.Counter(map(key, expected))
assert len(actual) == validation['assignmentRows']
assert len(new['accepted']) == validation['affectedRecords']

source = {row['ID']: row for row in csv.DictReader(open(data / 'cldf/forms.csv'))
          if row['Language_ID'] == 'Maliyad'
          and 'census2023tamilnadu' in row['Source']
          and row['Status'] not in ('entry','redirect')}
linked = {row['Child_ID'] for row in csv.DictReader(open(data / 'cldf/edges.csv')) if row['Rank'] == '1'}
linked.update(row['Form_ID'] for row in read_assignments() if str(row.get('Rank')) == '1')
linked &= set(source)
old = json.loads((p.parent / 'dravidian-surveys-20260921/proposed-decisions.json').read_text())
source_record = lambda x: x['record']['Language_ID'] == 'Maliyad' and 'census2023tamilnadu' in x['record']['Source']
prior_a = {x['record']['ID'] for x in old['accepted'] if source_record(x)}
prior_h = {x['record']['ID'] for x in old['held'] if source_record(x)}
new_a = {x['record']['ID'] for x in new['accepted']}
new_h = {x['record']['ID'] for x in new['held']}
sets = [linked,prior_a,prior_h,new_a,new_h]
assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
assert set(source) == set.union(*sets)
assert [len(x) for x in sets] == [0,37,11,len(new['accepted']),len(new['held'])]
assert len(source) == 498 and len(new_a) + len(new_h) == 450
report = dict(source='census2023tamilnadu',language='Maliyad',sourceRecords=len(source),
              priorLinked=len(linked),priorProposed=len(prior_a),priorHeld=len(prior_h),
              proposedRecords=len(new_a),heldRecords=len(new_h),unexaminedRecords=0,
              uniqueOwnership=True,manifestAgreement=True,assignmentRows=len(actual),validation=validation)
(p / 'checkpoint-001-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
coverage = {k: report[k] for k in ('source','language','sourceRecords','priorLinked','priorProposed','priorHeld','proposedRecords','heldRecords','unexaminedRecords','uniqueOwnership')}
(p / 'source-coverage-001.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
print({k: report[k] for k in ('sourceRecords','priorLinked','priorProposed','priorHeld','proposedRecords','heldRecords','assignmentRows','manifestAgreement')})
