import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass230';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5536-6',citation='CDIAL[5536.6]',evidence='Full CDIAL ḍala subsection 6, ḍhilla, explicitly gives Maithili ḍhīl louse. Kochila Tharu dhil/dhilə louse matches this regional family, preserving dental versus retroflex spelling, aspiration and final vowel. Use the specifically numbered louse-bearing branch 5536-6 rather than the general lump head or unaspirated branches. Local IA transmission remains unresolved. The longer Danuwar dhiluva forms are excluded pending their ending analysis.'),dict(parent='4828',citation='CDIAL[4828]',evidence='Full CDIAL cillaṭa/cillaḍa explicitly gives Hindi cīl(h)aṛ/cīl(h)ar and cillaṛ/cillar louse. Dang Tharu čilur/čɪlra/čɪlər/čilra fits that family with source vowel, syncope and rhotic notation retained. The dictionary itself supplies the louse sense, so no inference from a generic creeping-animal gloss is required. Regional IA transmission remains unresolved.')]
sets=[{'dʰil','dʰilə'}, {'čilur','čɪlra','čɪlər','čilra'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='louse':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
