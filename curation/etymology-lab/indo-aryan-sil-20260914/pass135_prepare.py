import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass135';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10507',citation='CDIAL[10507.2]',evidence='CDIAL 10507 subsection 2 explicitly supplies the eat sense, with Dameli žu-, Gawri and Kalasha žu-, and Khowar žib-. Turner tentatively explains Khowar b through expressive doubling in *yuvvati; this uncertainty is retained. Selected survey forms preserve their infinitive/finite endings and vowel length, and Dameli źīny ā retains the original spacing. The existing canonical node is 10507 (no separate subsection-2 node); its displayed yokes gloss does not exhaust the primary article. The distinct grain-eating ravate family at 10645 was compared and is not substituted. Local transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='eat' and r['Form'] in {'źībīk','źipe','źibīk','źībik','źūk','źūīk','źūnūs','źīny ā'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
