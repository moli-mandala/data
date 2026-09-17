import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass228';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4571',citation='CDIAL[4571.1]',evidence='Full CDIAL caṭaka subsection 1 gives Nepali caro bird/cari small bird, Kumaoni caṛo/caṛi and the -āka extension Bengali caṛāi, Oriya caṛāi and Assamese sarāi. The selected ca-/cə-/co-/tsa- survey bird forms follow this branch. Preserve source affricate and aspiration notation, schwa/o vowels and dental/retroflex/rhotic differences; the source bird sense is not narrowed to sparrow. Local IA transmission remains unresolved.'),dict(parent='4571-2',citation='CDIAL[4571.2]',evidence='Full CDIAL subsection 2 explicitly sets ciṭaka apart, with Prakrit ciḍiga bird, Hindi ciṛī/ciṛiyā bird, Bhojpuri ciraī and Awadhi ciraiyā. The selected Tharu/Danuwar ci-/tsi- bird responses match this front-vowel branch; retain source nasalization, affricate notation, final vowels and expanded feminine endings. The generic bird gloss remains unchanged, and local IA transmission is unresolved.')]
sets=[{'cari','carai','tsarai','cərā','cara','core','cəḍo','cʰərai'}, {'tsirai','cirəi','cire','čĩrĩyə̃','čirãⁱyã','čĩre','čĩrãⁱ','čiraĩyə̃'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='bird':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
