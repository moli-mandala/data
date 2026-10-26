"""Render pending Gondi review batches; no accepted-overlay writes."""
import collections,csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];LAB=P.parent;B=LAB/'Gondi';B.mkdir(exist_ok=True)
z=json.load(open(P/'proposed-decisions.json'));rows=z['accepted'];held=z['held'];articles=json.load(open(P/'primary-evidence.json'))
by=collections.OrderedDict()
for r in rows:by.setdefault(r['parent'],[]).append(r)
parents={}
with open(ROOT/'cldf/forms.csv') as f:
 for r in csv.DictReader(f):
  if r['ID'] in by:parents[r['ID']]=r
assert set(parents)==set(by)
start=max(int(f.stem.split('-')[1]) for f in B.glob('batch-*.json') if f.stem.split('-')[1].isdigit())+1
paths=[];index=['# Gondi LSI residual records — initial assessment complete','', '**Pending review. No new assignments have been saved. No rebuild has been run.**','',f'**{len(by)} proposed etyma** affect **{len(rows)} source records** with **{len(rows)} assignment rows**. **{len(held)} residual records** remain on explicit research hold; all **139** staged residuals have been assessed. An earlier checkpoint covered another **39 proposals and 4 holds** among the same 182 source records.','', 'The three previously questioned Gondi holds for ant/DEDR 4169, meat/DEDR 3373, and south-broom/DEDR 4794 are untouched. This pass adds no donor nodes or accepted sidecar rows.','', '| Review batch | Proposed etyma | Affected records |','|---|---:|---:|']
for bi in range(0,len(by),5):
 number=start+bi//5;chunk=list(by.items())[bi:bi+5];props=[]
 lines=[f'**Next {len(chunk)} proposals, ready for review. Nothing in this batch saved yet.**','',f'Gondi · LSI 1928 residual records · batch {number:03d}. Pending user review.','', '| # | Gondi | Proposed etymology | Evidence |','|---|---|---|---|']
 for j,(parent,items) in enumerate(chunk,1):
  article=articles[parent];link=f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={article["Page"]}'
  forms=list(dict.fromkeys((x['record']['Form'],x['record']['Gloss']) for x in items))
  label=', '.join('**'+f.replace('|','\\|')+'** ‘'+g.replace('|','\\|')+'’' for f,g in forms)
  explanations=list(dict.fromkeys(x['evidence'] for x in items))
  evidence=' '.join(explanations)
  lines.append(f'| {j} | {label} | Proposed reflex of [**DEDR {parent[1:]}**]({link}) | {evidence} ({len(items)} record{"s" if len(items)!=1 else ""}.) |')
  assignments=[dict(Form_ID=x['record']['ID'],Etymon_ID=parent,Kind=x['kind'],Rank='1',Status='proposed',Source=x['citation'],Notes=x['evidence'],Pos='') for x in items]
  props.append(dict(number=j,status='pending-review',parentId=parent,parentForm=parents[parent]['Form'],kind='reflex',citation=';'.join(dict.fromkeys(x['citation'] for x in items)),evidence=evidence,formIds=[x['record']['ID'] for x in items],records=[x['record'] for x in items],assignments=assignments))
 manifest=B/f'batch-{number:03d}.json';review=B/f'batch-{number:03d}-review.md';assert not manifest.exists() and not review.exists()
 manifest.write_text(json.dumps(dict(language='Gondi',batch=number,status='pending-review',date='2026-09-25',researchDirectory=str(P),proposals=props),ensure_ascii=False,indent=2)+'\n')
 lines+=['',f'Proposed assignment rows: **{sum(len(p["assignments"]) for p in props)}**. Affected records: **{sum(len(p["records"]) for p in props)}**. Relation kind: **reflex**. Ordinary inflection is kept under the same whole DEDR article; complete clauses and uncertain loans are excluded.','',f'Exact persistent IDs and source records: [{manifest.name}]({manifest.name}). See [research holds](../gondi-lsi-20260925/research-holds.json).']
 review.write_text('\n'.join(lines)+'\n');paths.append(str(manifest));index.append(f'| [{number:03d}](../Gondi/{review.name}) | {len(props)} | {sum(len(p["records"]) for p in props)} |')
index+=['','Candidate discovery used the existing homepage `fetchLemmaList` code and read-only database. Full DEDR articles were read from the cached primary-source audit. [Primary evidence](primary-evidence.json), [decisions](proposed-decisions.json), [holds](research-holds.json), and [coverage](record-coverage-001.json) are retained.','', 'All 182 compiled Gondi LSI source records are accounted for across prior pending proposals/holds and this pass. Current proposals and holds are research, not accepted links. No importer, rebuild, browser database refresh, commit, push, or deployment was run.']
(P/'REVIEW.md').write_text('\n'.join(index)+'\n');(P/'review-manifest-paths.json').write_text(json.dumps({'Gondi':paths},indent=2)+'\n')
print(json.dumps({'batches':len(paths),'etyma':len(by),'rows':len(rows),'held':len(held)}))
