"""Reconcile the initial unassigned queue with current compiled graph and overlays, without rebuilding."""
import csv,json,sys,datetime
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from etymology_assignments import read_assignments
P=Path(__file__).resolve().parent
initial=json.loads((P/'remaining-initial.json').read_text());wanted={r['ID'] for r in initial}
current={r['ID']:r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in wanted}
linked={r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1' and r['Child_ID'] in wanted}
linked|={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted' and r['Rank']=='1' and r['Form_ID'] in wanted}
remaining=[dict(r,**{k:current[r['ID']][k] for k in ['Language_ID','Form','Gloss','Source','Tags']}) for r in initial if r['ID'] in current and r['ID'] not in linked]
report={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initialQueue':len(wanted),'currentSurvivors':len(current),'currentlyLinkedInQueue':len(linked),'currentlyUnassignedInQueue':len(remaining),'missingIds':sorted(wanted-current.keys()),'scope':'The previously reconciled initial unassigned survey queue; not a new whole-workspace source inventory.'}
(P/'remaining-current.json').write_text(json.dumps(remaining,ensure_ascii=False,indent=1)+'\n');(P/'current-queue-reconciliation.json').write_text(json.dumps(report,indent=1)+'\n');print(report)
