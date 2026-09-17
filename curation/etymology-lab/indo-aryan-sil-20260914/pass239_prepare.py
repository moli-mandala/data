import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass239';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10539',citation='CDIAL[10539];tulpule1999[p. 579, entry 2]',evidence='Full CDIAL rakta explicitly gives blood; Tulpule Old Marathi p.579 entry 2 gives ragata blood and explicitly identifies Sanskrit rakta as its etymon, with reciprocal rakta/ragata comparison at p.578 entry 5. This directly documents the epenthetic voiced ragat type, rather than deriving it from a red-colour gloss alone. The selected survey rogat/rəgəṭ/rekeṭ/rekṭ/rakṭo/ṛakto/rəkhət/rakt a blood forms are grouped provisionally in this family. Source epenthetic vowels, stop voicing, aspiration, retroflexion and spacing remain unchanged. Learned or cross-IA transmission is unresolved; no claim of an Old Marathi loan to every lect is made.')]
forms={'rogat','rəgəṭ','rekeṭ','rekṭ','rakṭo','ṛakto','rəkhət','rakt a'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='blood' and r['Form'] in forms:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
a={'10539':json.loads((P/'pass238-primary-articles.json').read_text())['10539'],'tulpule1999':[dict(page=579,text='Raw source row: OM,10539,ragata,blood.,रगत; tulpule1999[p. 579, entry 2]; Etymology: Sk. rakta/cf. rakta; Entry_Key tulpule:p579:e2:v1. Reciprocal p.578 entry 5 rakta blood., Sk./cf. ragata. Source file data/other/forms/20260810-tulpule-old-marathi.csv.')]};(P/(stem+'-primary-articles.json')).write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare239.py').read_text());print('accepted',len(acc))
