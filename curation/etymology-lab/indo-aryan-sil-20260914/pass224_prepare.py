import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass224';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8042',citation='CDIAL[8042.1]',evidence='Full CDIAL pāḍḍa subsection 1 explicitly gives Prakrit paḍḍaya buffalo and pāḍī young buffalo, Gujarati pāḍɔ/pāḍī and Marathi pāḍā/pāḍī buffalo calf. The selected western paḍi/pāḍi/paḍo buffalo responses match this family with length and gender-ending variation retained. Preserve the generic survey gloss rather than changing it to calf. The dictionary proposes probable Dravidian origin; this is a link to the established IA lexical family, not a claim that the ultimate origin is settled. Distinguish the separately numbered pēḍḍa branch, and leave glottalized paṛʔi and o-vowel poḍ(i) for further review.'),dict(parent='9964',citation='CDIAL[9964]',evidence='Full CDIAL mahiṣa gives Maiya mhēsh, regional maīṣ/maĩš, Jaunsari mahiś, Mth mahis, Marathi mahīs/mhais and Kotgarhi mhɛś, as well as feminine bhaĩsi-type forms. The selected simple buffalo responses preserve vowel length/nasalization, palatal/retroflex fricative or affricate notation, and ordinary feminine/oblique vowel endings. Same-family mes/mhes and mes/mās slash responses are preserved intact and receive one family link. This does not settle local IA transmission. Ga-prefixed forms, unclear dorsal-x forms, m-less hesh forms and extra -niya material are excluded.')]
sets=[{'paḍi','pāḍi','paḍo'}, {'miheś','miheś / meś','maīẓ','maīeṣ','mãī̃ʦ̣','māīʦ̣','mheṣ','mahīs','mɦais','mæs','mæːs','maś','moyç','maçi','mẽːʃi','mẽʃi','ˈmẽʃi','mẽʃə','meʃi','bʰẽns','mãːʃi','meʃːi','bʰæsi','mes / mʰes','mes / mās'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='buffalo':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
