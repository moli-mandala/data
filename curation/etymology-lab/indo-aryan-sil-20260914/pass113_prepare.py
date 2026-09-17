import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass113';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
e='CDIAL 2528 addendum explicitly places Bengali kabe and Hindi kab when? with western Pahari kεbε, through remodeling of the ēvam ēva temporal-adverb family with interrogative ka. The selected kəbə/kebe/keb/kebey/kebəy/kob responses fit these regional kab/kebe forms. This is the dictionary’s analogical-family analysis, not unmodified inheritance from just so; vowel and final-glide variation and local Indo-Aryan transmission remain unresolved.'
rules=[dict(parent='2528',citation='CDIAL[2528, addendum]',evidence=e)];acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Form'] in {'kəbə','kebe','keb','kebey','kebəy','kob'} and r['Gloss'] in {'when','when?'}:acc.append(dict(record=r,parent='2528',family=0,kind='reflex',citation=rules[0]['citation'],evidence=e+' Exact response: '+r['Form']+'.'))
for name,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
