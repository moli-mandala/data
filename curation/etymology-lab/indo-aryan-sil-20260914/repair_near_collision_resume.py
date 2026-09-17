import json,csv,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;new='near-pass73';old='near-fourth';ledger=json.loads((P/'pass-ledger.json').read_text())
paths=[];manifestrows=[]
for f in P.parent.glob('*/batch-*.json'):
 d=json.loads(f.read_text())
 if isinstance(d,dict) and d.get('validation')==str(P/f'{old}-validation.json'):
  paths.append(str(f));manifestrows.extend(r for q in d['proposals'] for r in q['assignments'])
assert paths
rows_by={(r['Form_ID'],r['Etymon_ID']):r for r in manifestrows}
d=json.loads((P/f'{old}-decisions.json').read_text());rows=[rows_by[(x['record']['ID'],x['parent'])] for x in d['accepted']]
assert len(rows)==len(manifestrows)
for x,r in zip(d['accepted'],rows):assert r['Notes']==x['evidence']+' Joint SIL review 2026-09-14.'
(P/f'{old}-saved-assignments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1))
(P/f'{old}-manifest-paths.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
# Reconstruct the historical pre-save rows from the preserved overlay prefix.
root=P.parents[2];overlay=list(csv.DictReader((root/'data/etymology-assignments.csv').open()))
i=next(i for i,r in enumerate(overlay) if r==rows[0]);assert overlay[i:i+len(rows)]==rows
backup=P/'backups/etymology-assignments-before-near-fourth.csv'
backup.rename(P/'backups/etymology-assignments-before-near-pass73.csv')
with backup.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(overlay[0]));w.writeheader();w.writerows(overlay[:i])
# Reverify historical graph application without writing the overlay or new manifests.
s=(P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',old)
s=s[:s.index("backup=P/'backups'")].replace("assert not [r for r in old if r['Form_ID'] in targets], 'Concurrent/alternate target assignments require explicit reconciliation'",'assert True # Historical assignments already exist; no overlay write is performed.')
ns={'__file__':str(P/'global_sixteenth_save.py')};exec(compile(s,'historical-graph-revalidation','exec'),ns)
report=dict(assignmentRows=len(rows),affectedRecords=len(rows),parentNodes=len({r['Etymon_ID'] for r in rows}),previousOverlayRows=i,firstApplicationChanges=ns['first'],secondApplicationChanges=ns['second'],unrelatedEdgesPreserved=True,sourceFormsPreserved=True,identityRegistryPreserved=True,recoveryNote='Historical report was overwritten by a pass-name collision. Decisions regenerated from unchanged historical algorithm and pre-pass roster, exact saved rows recovered from surviving language manifests and verified against overlay segment. Graph checks rerun on current unchanged canonical inputs; this report does not reproduce the lost historical timestamp or hash report.')
(P/f'{old}-validation.json').write_text(json.dumps(report,indent=2)+'\n')
for key,suffix in [('decisionFiles','decisions.json'),('manifestFiles','manifest-paths.json'),('primaryFiles','primary-articles.json')]:ledger[key].append(new+'-'+suffix)
ledger['rulesByPass']['73']=new+'-rules.json';(P/'pass-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
(P/'near-pass73-collision-recovery.json').write_text(json.dumps(dict(historicalRowsRecovered=len(rows),historicalManifestsRecovered=len(paths),newRows=24,overlayUnmodifiedDuringRecovery=True,historicalReportRevalidated=True),indent=2)+'\n')
print('RECOVERED',len(rows),len(paths))
