import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass290';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10302',citation='CDIAL[10302]',evidence='Full megha explicitly means cloud and rain, with Prakrit meha, northern mẽ/mī̃, Kumaoni me and Gujarati mehulɔ/mevlɔ. The selected meg/megh/mek, nasal or reduced me forms and mevuḷo fit this family. Conservative velars may reflect learned or cross-IA transmission; preserve source deaspiration, devoicing, nasalization and vowels. The source cloud/rain distinction is maintained, not standardized.'),dict(parent='11567',citation='CDIAL[11567]',evidence='Full vārdala gives Prakrit vaddala cloud, Punjabi baddal, West Pahari badlu rain, Bengali bādlā rain, regional bādal/bādar and Gujarati vādaḷ/vādḷũ cloud. These support the selected b/d/l/r and western v-initial cloud/rain responses, including source retroflexion, gemination, vowels and ordinary endings. Local changes and cross-IA transmission remain qualified. Long multiword or mixed-family alternatives are excluded.')]
me={'mek','megh','megʰ','mʸeg','mẽgh','megʰ.o','mɛgh','meg','mẽg','mevuḷo','mẽ','mẽɣ','me'}
ba={'badal','badol','baḍal','baḍil','badūl / bādūl','bādal','badil','bəddəḷ','badul','badṛi','baḍːal','baḍːalə','baḍːalː','bʌdʌɾija','badʌṛ','vādəḷũ','vādəḷu','vādoḷu','vādelu','vādaḷu','vadalo','vaḍalo','uaḍlo','baḍalo','vaṭlo','vaḍəlo','vaḍaḷu','baḍalõ','vaḍala','bəḍ̚ri','bərri','bəḍri'}
sets=[('cloud',me),('rain',me),('cloud',ba),('rain',ba)]
rules=[rules[0],rules[0],rules[1],rules[1]]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare290.py').read_text());print('accepted',len(acc))
