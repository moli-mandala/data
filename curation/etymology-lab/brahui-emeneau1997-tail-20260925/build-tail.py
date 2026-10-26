"""Audit the six unresolved Emeneau 1997 source forms; no accepted writes."""
import collections,csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[2]
forms=[r for r in csv.DictReader((D/'cldf/forms.csv').open()) if r['Source'].startswith('emeneau1997brahui[')]
assert len(forms)==19
ids={r['ID'] for r in forms}
edges=collections.defaultdict(list)
for row in csv.DictReader((D/'cldf/edges.csv').open()):
 if row['Child_ID'] in ids:edges[row['Child_ID']].append(row)
prior={r['ID']:edges[r['ID']][0]['Parent_ID'] for r in forms if any(e['Rank']=='1' for e in edges[r['ID']])}
assert len(prior)==13
residual=[r for r in forms if r['ID'] not in prior]
assert len(residual)==6
notes={
 'puźźa':'Emeneau p. 443 §6 only tentatively compares this human-hair noun with DEDR 4477. DEDR 4477 has Tamil poccu “quantity of hair” and Kannada boccu “wool, down”, but the paper contrasts the regular Brahui pōs development of DEDR 4476. Keep the installed rank-2 hypothesis; no secure direct reflex is established.',
 'kūźing':'Emeneau p. 443 §6 weighs DEDR 2687 “shrink, contract” against DEDR 1876 “recoil, shyness”. The former better fits initial k-; neither explains Brahui ž. Both are already ranked hypotheses (2 and 3), not accepted etymologies.',
 'pisfing':'Emeneau p. 443 §7 calls DEDR 4135 “knead, squeeze” simpler than DEDR 4183 “wring, squeeze”, but -f- remains unexplained even under the preferred analysis. Existing ranks 2 and 3 preserve both hypotheses.',
 'śupping':'Emeneau p. 444 §8.1 presents the DEDR 2621b “suck, sip” comparison tentatively. Expressive reshaping and a possible Indo-Aryan comparison prevent a categorical assignment; keep its rank-2 hypothesis.',
 'taṛifing':'The source separates this “turn sour (milk)” sense from homographic taṛifing “be slaughtered”, which is securely assigned to DEDR 3029. Emeneau p. 444 §11 explicitly leaves the sour-milk sense unetymologized.',
 'dūī':'Emeneau p. 446 §15 n. 6 finds CDIAL 5228 tempting for “tongue” but leaves the reconstruction and immediate borrowing source unresolved. The rank-2 CDIAL hypothesis is retained; the source variant duvī has an existing rank-1 lexical link to dūī, not a resolved etymon.',
}
assert {r['Form'] for r in residual}==set(notes)
held=[{'record':r,'reason':notes[r['Form']],'status':'research-hold','existingHypotheses':[
 {'parent':e['Parent_ID'],'rank':e['Rank'],'kind':e['Kind']} for e in edges[r['ID']]]} for r in residual]
articles={}
want={'d4477','d4476','d2687','d1876','d4135','d4183','d2621','d3029'}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
 for a in csv.DictReader(fh):
  if a['Entry_ID'] in want:articles[a['Entry_ID']]={k:a[k] for k in ('Page','Citation','Raw_Article')}
assert set(articles)==want
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('survey-inventory.json',residual)
write('proposed-decisions.json',{'status':'pending-review','accepted':[],'held':held})
write('research-holds.json',held)
write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
write('review-manifest-paths.json',[])
write('source-coverage-001.json',{'source':'emeneau1997brahui','totalCompiledRecords':19,
 'priorRankOneAssignedRecords':13,'newProposedRecords':0,'currentHeldRecords':6,'unexaminedRecords':0,
 'uniqueOwnership':True,'recordOwnership':{r['ID']:('prior-rank-one:'+prior[r['ID']] if r['ID'] in prior else 'current-hold') for r in forms},
 'priorRankOne':prior,'currentHolds':[r['ID'] for r in residual]})
print(json.dumps({'source':19,'priorRankOne':13,'newProposals':0,'holds':6,'primaryArticles':len(articles)}))
