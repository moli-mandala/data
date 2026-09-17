import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass223';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9153',citation='CDIAL[9153]',evidence='Full CDIAL barkara gives regional bakarā/bakrā and feminine bakarī/bakrī goat, plus Garhwali/Kumaoni/Nepali bākhro with aspiration. Selected Tharu bakariya-type responses retain the ordinary expanded feminine ending, and Kului a/schwa plus retroflex-r forms retain their exact articulation and vowel notation. This is the barkara family; rounded forms overlapping bokka are excluded. Local IA transmission remains unresolved.'),dict(parent='9312',citation='CDIAL[9312]',evidence='Full CDIAL bokka explicitly gives Prakrit bokkaḍa goat, Old Gujarati bokaḍaü he-goat and Marathi bokaḍ; it also compares Sanskrit bukka goat and gives Oriya bukā. Selected western Bhil/Bareli bukaḍo/bukəḍi/bukuḍo/bukḍo-type goat forms have the documented stop extension, with ordinary gender/vowel endings retained. The survey generic goat sense is preserved rather than narrowed to male. The u/o variation is qualified; no particular cross-IA donor route or deeper origin of bokka is asserted. Rhotic forms that overlap barkara remain separate.')]
sets=[{'bʌkʌɾija','bakʰra','bəkəɽə','bəkəɽi','baːkəɽə','bʌkəɽə','bəkəɽo','bəkhəria'}, {'bukaḍo','bukəḍi','bukaḍi','bukuḍo','bukḍo','bukḍi','bakḍi','bokaḍu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='goat':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='goat' and r['Form'] in {'bukuṛi','bukəṛi','bukhṛo','bokṛa','bokɽe','bokɽə','bokəɽə','bokəɽu'}:
  held.append(dict(record=r,families=[0,1],reason='Rounded rhotic goat form overlaps barkara 9153 and bokka 9312. Full 9153 explicitly reports Hindi bokrā as crossed with bokka, while 9312 independently lists Mth/Hindi bokṛā/bokrā and Gujarati bokṛũ. Local source morphology/phonology is needed to choose the parent for this record; the hold is not just uncertainty about IA borrowing.',passNumber=223))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc),'held',len(held))
