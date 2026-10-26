import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'rezaibaghbidi2003zargari[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
parents={'drink':'f_hqru4n3h67n7w','run':'f_73nygaykqt5ka','cut':'f_ldvlw2qaq5ix6','eat':'f_l5cg5m5rzv2ks'}
a=[];h=[]
for r in rs:
 if r['ID'] in assigned:continue
 if r['Form'] in ['piyay','piyayipey','piyāvd']:
  a.append(dict(record=r,parent=parents['drink'],kind='derived',citation=r['Source']+';rezaibaghbidi2003zargari[p. 141, section 3.15.1];rezaibaghbidi2003zargari[p. 142, section 3.17.3.1]',evidence='The primary grammar explicitly derives causative piy-ay- make drink from piy- drink with -ay (< *-av), and identifies piy-āvd- as its perfect stem and piyayipey as its infinitive. Link all three ordinary forms of this derived causative to the existing same-source base piy (f_hqru4n3h67n7w); its accepted compiled ancestry is pibati8209. This records a source-defined synchronic derivation, not direct inheritance of the entire causative from Sanskrit.'))
 for fs,key,reason in [(['nāśay','nāśayipey','nāśāvd'],'run','Source3.15.1 explicitly forms the causative with -ay and perfect -āvd.'),(['cʰindiv','cʰindil'],'cut','Source3.16.1 explicitly adds mediopassive -iv/-il to the perfect stem čhind-.'),(['xaliv','xalil'],'eat','Source3.16.1 explicitly adds mediopassive -iv/-il to the perfect stem xal-.')]:
  if r['Form'] in fs:h.append(dict(record=r,parent=parents[key],kind='derived',reason=reason+' The parent has accepted overlay ancestry but remains compiled-unlinked; current etymology_lab.py rejects that parent at its status check. No save bypass or rebuild.'))
assert len(a)==3 and len(h)==7,(len(a),len(h))
(P/'pass030-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
locs={s for x in a+h for s in x['record']['Source'].split(';')};aud=[r for r in csv.DictReader(open('data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv')) if r['Source'] in locs];(P/'pass030-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
(P/'pass030-review.md').write_text('''# Zargari: explicit causative derivation

3 records / 3 derived rows, one same-source base. Seven further source-explicit derived records remain pending over the shared saver’s compiled-parent status check.

| # | Zargari | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **piyay / piyayipey / piyāvd** ‘to make drink’ | Derived from Zargari **piy** ‘drink’, existing **f_hqru4n3h67n7w**, ultimately pibati8209 | Printed p141,3.15.1 explicitly defines causative -ay (< *-av), perfect -āvd and this example; p142,3.17.3.1 defines infinitive -ipey. The source base is already linked in the compiled graph. All three forms retain their source morphology. |

Primary grammar: zargari-primary-pages.json (read in pass028); source audit rows copied to pass030-source-audit.json. The stem’s historical pibati ancestry was reviewed in pass029; this batch’s evidence for derivation is the grammar itself, not word similarity.

Pending exact derivations are recorded in decisions held[]: nāśay/nāśayipey/nāśāvd → nāś; čhindiv/čhindil → čhind; xaliv/xalil → xal. These parents have accepted sidecar ancestry but compiled Status=unlinked. The shared saver still rejects them before scratch validation. No helper change, direct sidecar edit or rebuild was performed. The older čhiv→čhib variant limitation is the same issue.

Validation and save status: pending.
''')
print(len(a),'accepted',len(h),'held')
