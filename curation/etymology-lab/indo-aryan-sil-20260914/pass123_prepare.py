import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass123';assert not (P/(stem+'-decisions.json')).exists()
spec=[({'hardul','hardūl','hardol','haldal','hardeḷ','halder','aldar','əldər'},'CDIAL 13992.1 haridrā explicitly gives Awan hàrdul, Lahnda hardal/haladr/haldhar and Gujarati haḷdar turmeric. The selected hardul/hardol/haldal/hardeḷ/halder and h-less aldar/əldər fit this rhotic/lateral family with source vowel and h variation preserved.'),({'haḷid','halid','hoḷid','oḷid','həlid','hoḷed','haḷed','hoḷdi','oldi','holdi','halit'},'CDIAL 13992.1 haridrā gives Gujarati/Marathi haḷad, Oriya haḷadi and Hindi/Punjabi haldī turmeric. Selected haḷid/halid/hoḷid/oḷid/həlid/hoḷed/haḷed/hoḷdi/oldi/holdi/halit preserve regional vowel raising/rounding, h loss, metathesis and final devoicing notation. These are family matches with those qualifications, not claims that the dictionary quotes each survey variant.')]
rules=[dict(parent='13992',citation='CDIAL[13992.1]',evidence=e+' The Assamese turmeric-coloured hāridra subsection is not selected. Local Indo-Aryan transmission remains unresolved.') for ss,e in spec]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='turmeric':continue
 for i,(ss,e) in enumerate(spec):
  if r['Form'] in ss:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
