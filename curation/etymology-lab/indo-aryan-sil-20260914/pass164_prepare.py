import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass164';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9188',citation='CDIAL[9188]',evidence='CDIAL 9188 bahukāra explicitly gives Prakrit baühārī/bohārī brush and Punjabi/Lahnda bahārī/bohārī/buhārā broom, with Sindhi bauārī in the addendum. Selected bahari/bairi/buar responses match this documented broom family. Vowel contraction, aspiration migration or loss, glide realization and inflection remain qualified and unchanged; the link does not settle local IA transmission. Extra-n formations and highly contracted forms are excluded.')]
sets=[{'bairi','bāyri','bhahari','bɦahari','bāyro','beheri','baheri','bɦahəri','bayri','bɦāiri','boari','bʷari','bʰʷari','bʰuːaːr','bʰwaːr','bʊaːr','ˈbʰu(h)ar','bʊ(h)aːr','bʊwar','bhāir','bhāiri'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='broom':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
