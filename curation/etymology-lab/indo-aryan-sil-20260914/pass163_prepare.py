import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass163';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11072-2',citation='CDIAL[11072.2]',evidence='CDIAL 11072 subsection 2 *lugga explicitly gives Nepali/Oriya luga cloth, Hindi lugra rags, Gujarati lugrũ clothes and Marathi lugḍẽ. The selected lug/lugəḍ/lugər cloth forms fit that documented branch; vowels, rhotic/retroflex notation and endings are retained and qualified. Turner doubts direct derivation from broken MIA forms and leaves the deeper defective group uncertain; local transmission is unresolved.'),dict(parent='2871',citation='CDIAL[2871]',evidence='CDIAL 2871 karpaṭa explicitly gives Kashmiri kapur cotton cloth, Sindhi kapaṛu and widespread kapaṛ/kapṛa cloth forms. Selected kapur/kepeḍa/kapəṛu responses preserve this family, with vowel variation and dental/retroflex-r notation qualified. Turner discusses competing deeper origins and explicit IA borrowing; the link does not settle those issues or local transmission.')]
sets=[{'luɡʌɾa','lʊɡʌɽa','lu.ga','lugəḍu','lugḍu','lugaḍa','lugəḍo','lugaḍo','lugəra','lugərə'}, {'kapur','kaprāī','kepeḍa','kepera','kapəṛu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cloth':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
