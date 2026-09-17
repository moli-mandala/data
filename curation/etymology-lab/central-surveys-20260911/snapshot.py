"""Read-only corpus snapshot; never writes CLDF, identities or accepted overlay."""
import csv,json,collections,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[3]; out=Path(__file__).resolve().parent
sources={'Malvi':'varghese-john-samuel2009malvi','Nimadi':'vunnamatla-john-samuvel2012nimadi','Bagheli':'koshy2022bagheli'}
forms=list(csv.DictReader(open(root/'cldf/forms.csv')))
assert all(r['ID'].startswith('f_') for r in forms if any(s+'[' in r['Source'] for s in sources.values()) and not r['Redirect']), 'Intermediate build has nonpersistent target IDs; retry later'
linked={r['Child_ID'] for r in csv.DictReader(open(root/'cldf/edges.csv')) if r['Rank']=='1'}
linked|={r['Form_ID'] for r in csv.DictReader(open(root/'data/etymology-assignments.csv')) if r['Rank']=='1' and r['Status']=='accepted'}
summary={}
for name,source in sources.items():
 rows=[r for r in forms if not r['Redirect'] and source in {s.split('[')[0].strip() for s in r['Source'].split(';')}]
 remaining=[r for r in rows if r['ID'] not in linked]
 (out/f'{name}-inventory.json').write_text(json.dumps(remaining,ensure_ascii=False,indent=2))
 gs=collections.defaultdict(list)
 for r in remaining:gs[r['Gloss']].append(r)
 (out/f'{name}-gloss-inventory.txt').write_text('\n'.join(g+' | '+str(len(rs))+' | '+'; '.join(dict.fromkeys(r['Form'] for r in rs)) for g,rs in gs.items())+'\n')
 summary[name]={'source':source,'total':len(rows),'remaining':len(remaining),'language_ids':sorted({r['Language_ID'] for r in rows}),'distinct_glosses':len(gs),'snapshot_sha256':hashlib.sha256((root/'cldf/forms.csv').read_bytes()).hexdigest()}
(out/'inventory-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
