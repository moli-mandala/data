import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass137';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6909',citation='CDIAL[6909]',evidence='CDIAL 6909 *nakka nose explicitly gives Prakrit ṇakka, Awan nak, Punjabi nakk, Nepali/Bengali nāk, Oriya nāka and Gujarati nāk. Selected ṇak/ṇake and nako/nakho forms match this short nose family; survey initial retroflex notation, final vowels and aspiration are preserved as regional qualifications. Initial ṇ is not claimed to prove unchanged Prakrit retention. Longer suffixal and nath- formations remain separate, and local Indo-Aryan transmission is unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='nose' and r['Form'] in {'ṇak','ṇake','nako','nakʰo','nakʰə'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
