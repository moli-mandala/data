import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass175';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='3153',citation='CDIAL[3153.1]',evidence='Full CDIAL *kicca section 1 explicitly gives Bengali kicaṛ, Gujarati kīcaṛ, Marathi kicaḍ and Sindhi Kachchhi kicaḍ, while Phalura kičal/číčal illustrates affricate variation in the family. Selected kicar/kicaḍ forms preserve the k-initial mud stem and documented rhotic/retroflex extension. Affricate ts notation and aspiration/vowels remain qualified; local IA transmission is unresolved. Reordered cik- and differently vocalized koc- forms are excluded pending comparison with the cross-referenced *cikka family.'),
 dict(parent='2869',citation='CDIAL[2869]',evidence='Full CDIAL kardama gives Oriya kādua/kādo, Gujarati kādav, Maithili kādau/kadwā and Hindi kādau/kādo, alongside nasalized Old Awadhi kāṁdau and Hindi kā̃daw/kā̃do from *kaddaṽa. These support the selected short kādev/kaduə/kaḍeu and nasal variants with vowel and dental/retroflex notation retained. Extra -ḍu suffix forms are excluded; local IA transmission and the article’s reported paṅka influence remain unresolved.')]
sets=[set('kicar kitsar kicoḍ kicəḍu kichaḍ kicaḍo kicər kiṭsar'.split()),set('kaduə keṇḍəu kaḍeu kaṇḍe kaṇḍeu kaṇḍəu kādəv kaḍu'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='mud':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
