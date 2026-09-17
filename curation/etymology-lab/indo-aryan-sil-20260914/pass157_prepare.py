import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass157';assert not (P/(stem+'-decisions.json')).exists()
url=json.loads((P/'pass157-molesworth-research.json').read_text())[0]['url']
rules=[dict(parent='12607',citation='CDIAL[12607]',evidence='CDIAL 12607 śepyā explicitly connects Pali/Prakrit cheppā/cheppa/chippa tail with Marathi śep/śẽp. Molesworth pp. 800–801 explicitly gives śepaṭī/śempaṭī/śempaḍī tail and derives śepūṭ from śep. The survey sepṭ/sempṭ/sepḍ forms fit this documented western tail series; aspiration, sibilants, nasal spelling and endings are qualified. This identifies the documented lexical family without inventing an ancient ṭ suffix. Local IA transmission is unresolved. Primary Molesworth response: '+url),dict(parent='12607',citation='CDIAL[12607]',evidence='CDIAL 12607 documents both ch- and ś- tail forms, including Prakrit cheppa/chippa and Marathi śẽp. Molesworth pp. 800–801 lists nasal śempaṭī/śempaḍī beside śepaṭī. Western cemṭ/chemṭ/semṭ/cimṭ forms are compared with that nasal extended series, with loss of the labial stop in the nasal cluster and regional vowel, aspiration and dental/retroflex notation explicitly qualified. These links identify the supported family; exact local phonological history and transmission remain unresolved. Primary response: '+url)]
sets=[{'śepaṭi','sepḍo','sempṭi','sepṭo','sapṭa','cepa','çepṭi'}, {'chemṭo','chemṭi','cemṭo','cehmṭo','cimṭũ','semṭo','cemṭā','ceməṭā','ṣemṭi','semṭi','camṭi','chamṭi','cmṭā','semti','semto','chemto','chemti'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='tail':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
