import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass148';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='13067',citation='CDIAL[13067];platts1884[663]',evidence='CDIAL 13067 sakāla has early-in-the-morning sense and Oriya saaḷa early. Platts 663 explicitly gives sa-kāl/su-kāl dawn, sakālī early morning, and sakār/sakārī/sakāre as equivalents. The survey l/ḷ/r forms and regional o/a vowels thus match the morning family; aspiration, s/ś/h and retroflex-liquid spellings are retained and qualified. The precise regional transmission is unresolved.'),dict(parent='11813',citation='CDIAL[11813]',evidence='CDIAL 11813 *vibhāna explicitly gives Kumaoni byān, Nepali biyāna, eastern bihān and WPah. bhèṇi dawn. It also allows an alternative MIA vibhāyana derivation. Short ben/bhen and byan forms fit the documented contracted dawn family; contraction and local transmission remain qualified, and the deeper alternative is not resolved.'),dict(parent='11813a',citation='CDIAL[11813a];platts1884[193]',evidence='CDIAL 11813a *vibhāniḥsāra explicitly supplies bhinsar/bhinsār dawn and bhyānsar. Platts 193 also records bhinusār beside bhinsār. The survey bhunsar/bunsar series is compared with the latter u-bearing variant, with contraction, vowel length, nasal place and deaspiration explicitly qualified. Final vowels are retained. This is the extended dawn family, not the shorter *vibhāna node; local transmission remains unresolved.')]
sets=[{'sʌkaɾe','səkal','sokal','sakal','hākʰḷī','hakʰḷe','hakaḷe','sākār','sākārī','sakār','sakārɛ','sakʰāri','sʌkoɾ','sʌkaɾ','sokale','sukal','səkaḷa','sʌkṛa','sʌkaṛ','sʌkale','sʌkṛeya','sʌkṛiya','śokal','ʃɔkal','sekal','səkare','sakkale'}, {'bʰen','ben','bʸana','bʸan'}, {'bʰunsāræ','būnsāro','bunsāro','bʰūmsāra','bʰūnsāro','bʰūnsārā','bʰūnsarā','bʰunsārɛ','bʰinsər'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='morning':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print(len(acc));print([(q['parent'],sum(a['family']==i for a in acc)) for i,q in enumerate(rules)])
