import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass156';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8249-2',citation='CDIAL[8249]',evidence='The full CDIAL 8249 prose explicitly separates a -ḍa extension, represented in Jambu by 8249-2 *pucchaḍa: Sindhi puchaṛu/puchiṛī, western Pahari põchaṛ, Nepali puchar and Gujarati puchṛũ tail. The selected pucar/pucəḍ/puciṛ forms fit that extended branch; aspiration, affricate notation, dental/retroflex rhotics and endings remain qualified and unchanged. Local IA transmission remains unresolved. Citation is to the entry prose, not its unrelated numbered nasal addendum.'),dict(parent='8249',citation='CDIAL[8249]',evidence='CDIAL 8249 explicitly records the -la forms L. pūchal, Awankari puchul and Hindi pū̃chlā tail. The survey pucul/pūcal forms match that documented extension; vowel length, aspiration and retroflex-l notation remain qualified. No separate -la parent exists in the canonical nodes inspected, so the documented base node is used; local IA transmission is unresolved.'),dict(parent='8249',citation='CDIAL[8249]',evidence='CDIAL 8249 documents simple puch/pūch tail and nasal puṁcha/pū̃ch forms in the full prose and addendum. The short pucche/phučhi/pũči and Kullui puṇch forms match this family with nasal, aspiration and final-vowel notation retained and qualified. Pucchin 8250 denotes tailed animals and is not automatically substituted for every final-i tail form; local transmission remains unresolved.')]
sets=[{'pucaṛī','pūciṛī','puśiṛī','pucʰari','pucəri','pucār','puccʰar','pucəḍu','pusḍu','pucḍu','pucuḍu','putsaḍu','pucəḍi','pucəṛo','pucaṛ','puceṛo','pužar','pucʰoṛ'}, {'pūcal','pūcaḷ','pʰūcal','pucul'}, {'puccʰe','phučhi','pũči','pʊɳtʃə','pʊɳtʃʰ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='tail':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
