"""Read-only reconciliation of the earlier survey scope with current CLDF and sidecars."""
import csv, json, sys, hashlib, collections
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sys.path.insert(0,str(ROOT))
from etymology_assignments import read_assignments, assignment_files
OLD=P.parent/'indo-aryan-sil-20260914'
excluded=set(json.loads((OLD/'scope-correction.json').read_text())['excludedIds'])
old={r['ID']:r for r in json.loads((OLD/'inventory.json').read_text()) if r['ID'] not in excluded}
overlay=read_assignments()
linked={r['Form_ID'] for r in overlay if r['Status']=='accepted' and r['Rank']=='1'}
rejected={(r['Form_ID'],r['Etymon_ID']) for r in overlay if r['Status']=='rejected'}
for r in csv.DictReader((ROOT/'cldf/edges.csv').open()):
    if r['Rank']=='1' and (r['Child_ID'],r['Parent_ID']) not in rejected: linked.add(r['Child_ID'])
langs={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
remaining=[]; found=set()
for r in csv.DictReader((ROOT/'cldf/forms.csv').open()):
    if r['ID'] not in old: continue
    found.add(r['ID'])
    if r['ID'] not in linked and not r.get('Redirect'):
        remaining.append({k:r.get(k,'') for k in ['ID','Language_ID','Form','Gloss','Source','Tags','Description']}|{'clade':langs.get(r['Language_ID'],{}).get('Clade','')})
(P/'remaining-initial.json').write_text(json.dumps(remaining,ensure_ascii=False,indent=1)+'\n')
counts={'scope_records':len(old),'current_scope_records':len(found),'missing_old_ids':sorted(set(old)-found),'remaining_records':len(remaining),'languages':dict(collections.Counter(r['Language_ID'] for r in remaining).most_common()),'sources':dict(collections.Counter(r['Source'].split('[')[0] for r in remaining).most_common())}
(P/'inventory-summary.json').write_text(json.dumps(counts,ensure_ascii=False,indent=1)+'\n')
paths=[ROOT/'cldf/forms.csv',ROOT/'cldf/edges.csv',ROOT/'data/form-identities.csv',*assignment_files()]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
(P/'initial-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in paths},indent=1)+'\n')
print(json.dumps(counts,ensure_ascii=False,indent=1))
