"""Read-only placement and identity reconciliation for this run's rows."""
import csv,json,sys,re
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from etymology_assignments import read_assignments,SidecarResolver,FIELDS
P=Path(__file__).resolve().parent
corrections=[json.loads(f.read_text()) for f in P.glob('correction*-retraction.json')]
def removed_for(ledger):
 return {i for c in corrections if Path(c['originalLedger']).name==Path(ledger).name for i in c['formIds']}
new=[r for f in P.glob('pass*-saved-assignments.json') for r in json.loads(f.read_text()) if r['Form_ID'] not in removed_for(f)]
ids={r['Form_ID'] for r in new};resolver=SidecarResolver()
allrows=read_assignments();own=[r for r in allrows if r['Form_ID'] in ids]
signature=lambda rows:Counter(tuple(r.get(k,'') for k in FIELDS) for r in rows)
assert signature(new)==signature(own), 'current rows differ from active save ledgers'
source_keys=set(re.findall(r'@\w+\{([^,\n]+)',Path('cldf/sources.bib').read_text()))
unknown_sources=sorted({c.strip().split('[',1)[0] for r in new for c in r['Source'].split(';') if c.strip()}-source_keys)
assert not unknown_sources, unknown_sources
mis=[dict(row=r,actual=str(r.path),expected=str(resolver.path_for(r['Form_ID']))) for r in own if r.path!=resolver.path_for(r['Form_ID'])]
linked={r['Form_ID'] for r in allrows if r['Status']=='accepted' and r['Rank']=='1'}
with open('cldf/edges.csv') as f:
 rd=csv.DictReader(f);print('edge fields',rd.fieldnames)
 for r in rd:
  if r.get('Target_ID')=='f_hdsesne6llqf2' or r.get('Source_ID')=='f_hdsesne6llqf2':print(r)
extra=[]
with open('cldf/forms.csv') as f:
 for r in csv.DictReader(f):
  if r['ID']=='f_hdsesne6llqf2':extra.append(dict(record=r,hasAcceptedSidecar=r['ID'] in linked))
(P/'new-row-placement-audit.json').write_text(json.dumps(dict(savedRows=len(new),sidecarRows=len(own),misplaced=mis,exactLedgerMatch=True,unknownSourceKeys=unknown_sources),ensure_ascii=False,indent=1)+'\n')
(P/'supplemental-alias-record.json').write_text(json.dumps(extra,ensure_ascii=False,indent=1)+'\n')
print('new rows',len(new),'own current rows',len(own),'misplaced',len(mis),'extra',[(x['record']['ID'],x['hasAcceptedSidecar']) for x in extra])
assert len(new)==len(own) and not mis
