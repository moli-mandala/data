"""Render the review-only Muduga tail; never writes accepted sidecars."""
import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;LAB=P.parent;DATA=P.parents[2]
d=json.loads((P/'proposed-decisions.json').read_text());rows=d['accepted'];held=d['held']
assert d['status']=='pending-review' and len(rows)==2 and len(held)==8
ids={r['ID'] for r in json.loads((P/'survey-inventory.json').read_text())}
assert ids=={x['record']['ID'] for x in rows+held}
parent={}
for r in csv.DictReader(open(DATA/'cldf/forms.csv')):
 if r['ID'] in {x['parent'] for x in rows}:parent[r['ID']]=r
assert len(parent)==2
dest=LAB/'Muduga';dest.mkdir(exist_ok=True)
manifest=dest/'batch-029.json';review=dest/'batch-029-review.md'
if manifest.exists():
 old=json.loads(manifest.read_text());assert old.get('researchDirectory')==str(P)
props=[];lines=['**Next 2 proposals, ready for review. Nothing in this batch saved yet.**','','Muduga · Arsenault–Abraham 2022 residual records · 2026-09-25 · batch 029. Pending user review.','','| # | Muduga | Proposed etymology | Evidence |','|---|---|---|---|']
for i,x in enumerate(rows,1):
 r=x['record'];pid=x['parent'];page={'d2484':'217','d3111':'270'}[pid]
 proposed=f'Proposed reflex of [**DEDR {pid[1:]}**](https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={page})'
 label=f"**{r['Form']}** ‘{r['Gloss']}’"
 lines.append(f"| {i} | {label} | {proposed} | {x['evidence']} (1 record.) |")
 assignments=[dict(Form_ID=r['ID'],Etymon_ID=pid,Kind='reflex',Rank='1',Status='proposed',Source=x['citation'],Notes=x['evidence'],Pos='')]
 props.append(dict(number=i,status='pending-review',parentId=pid,parentForm=parent[pid]['Form'],kind='reflex',citation=x['citation'],evidence=x['evidence'],formIds=[r['ID']],records=[r],assignments=assignments))
manifest.write_text(json.dumps(dict(language='Muduga',batch=29,status='pending-review',date='2026-09-25',researchDirectory=str(P),proposals=props),ensure_ascii=False,indent=2)+'\n')
lines+=['','Proposed assignment rows: **2**. Affected records: **2**. Relation kind: **reflex**. The other **8** source records are documented research holds, including an unresolved loan, compounds, a clause, and a DEDR number collision.','',f'Exact persistent IDs and complete source records: [{manifest.name}]({manifest}). See the [research holds](../muduga-arsenault-tail-20260925/research-holds.json).','','No new lexical material or donor nodes are introduced. No accepted assignment or rebuild has been run.']
review.write_text('\n'.join(lines)+'\n')
(P/'REVIEW.md').write_text('# Muduga Arsenault–Abraham residual source review\n\n**Pending review. No assignments saved, and no rebuild run.**\n\nTwo pending DEDR-family proposals are in [Muduga batch 029](../Muduga/batch-029-review.md); eight records are documented holds. All ten residual records from the 109-record source are accounted for. The [primary DEDR article evidence](primary-evidence.json), [homepage searches](discovery-results.json), [comparanda searches](comparanda-results.json), and [decisions](proposed-decisions.json) are preserved.\n')
print('rendered',manifest)
