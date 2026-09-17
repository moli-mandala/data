import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass158';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11500',citation='CDIAL[11500]',evidence='CDIAL 11500 vātālī explicitly supplies Old Awadhi bayāri and Awadhi bayār wind. Selected bayar/byar/beyar responses match that extended wind family, with vowel contraction and y/h alternation in the eastern forms qualified. The entry discusses and rejects an alternative vārdala explanation; this link follows its vātālī analysis. Local transmission remains unresolved.'),dict(parent='11544',citation='CDIAL[11544]',evidence='CDIAL 11544 vāyu explicitly supplies Prakrit vāu, Punjabi/Assamese bau/bāu, Bengali bāo and eastern bāū. Selected bāyāū/bau/bou and vāyū preserve that documented family, with vocalic notation retained. Turner warns that vāyu and vāta are not distinguishable in every modern form: the link follows explicit comparanda without claiming that warning is resolved for each local transmission.'),dict(parent='7978',citation='CDIAL[7978]',evidence='CDIAL 7978 pavana wind explicitly supplies Prakrit pavaṇa/payaṇa and modern pavaṇ/pauṇ. Eastern pobon/pəbən responses preserve the fuller pavana stem, with v/b and vowel realization qualified; the relation identifies the family without deciding learned reinforcement or local IA borrowing.')]
sets=[{'bɛyar','biaɾ','bajaɾ','bʌjaɾ','bjaɾ','bayar','bəⁱyar','bʸar','baⁱyar','baⁱyal','bʌjal','behar','bɛhɛr','bahar','behɛr','bɛhar'}, {'vāyū','bāyāū','bau','bou̯'}, {'pəbən','pəu̯bən','pəbənə','pobon'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='wind':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
