import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass154';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9530-7',citation='CDIAL[9530.7]',evidence='CDIAL 9530 subsection 7 *bhuṇḍa explicitly gives Gujarati bhũḍũ bad and Hindi bhũḍā ugly. The straightforward western Bhil/Marwari bhunḍ/bhuṇḍ bad forms match that specific family. Nasal transcription, vowel length and adjectival endings are retained and qualified. The separate pig/belly entries 9551/9531 and dental *bhunda 9532 were compared, not substituted. Local IA transmission remains unresolved.'),dict(parent='9754-5',citation='CDIAL[9754.5]',evidence='CDIAL 9754 subsection 5 manda explicitly means bad and supplies Sindhi mando, Awankari mandā bad and Punjabi mandā. The simple Gojri/Pothwari/Awankari mando/manda/mandī responses match this branch. Inflection and local transmission are qualified; multiword alternatives are not reduced to this component.'),dict(parent='9289',citation='CDIAL[9289]',evidence='CDIAL 9289 *bura explicitly gives western Pahari buro/burā bad and Sindhi buro. Kullui buɽə forms fit the western Pahari bad family; the survey retroflex rhotic and vowel notation remain qualified and preserved. This does not assign the separate boro response to the same subsection or settle local borrowing.')]
sets=[{'bʰūṇḍɔ','bɦunḍũ','bɦuṇḍo','bɦunḍo','bɦuṇḍā'},{'mando','manda','mandī'},{'buˈɽə','bʊɽə','ˈbuɽə'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='bad':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
 else:
  if r['Form'] in {'bʰuṇḍa','buṇḍa','bʰunḍa'}:held.append(dict(record=r,families=[],reason='The northern Gojri/Kaithal bhunḍa bad forms need a subsection decision: CDIAL 9530.5 *bhuṇṭa explicitly yields Punjabi bhuṇḍā ugly, while 9530.7 *bhuṇḍa gives Hindi ugly and Gujarati bad. Surface ḍ alone cannot decide. The broader defective family is plausible, but the exact reconstructed parent remains unresolved.',passNumber=154))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc),'held',len(held))
