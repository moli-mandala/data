import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass255';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8047',citation='CDIAL[8047.1]',evidence='Full pāṇḍara subsection 1 explicitly gives Bshk pΛṇar/pΛna, Chil pΛnaro, Mai panara, Palula paṇāru and Oriya pāṇḍarā white. The selected northern panar/panāro/pānḍaro forms match that documented branch. Source vowels, retroflexion, rhotic/lateral notation and final inflection remain intact. The a-stem regional comparison supports this assignment rather than assuming the separate pāṇḍura subsection 2; local development and cross-IA transmission remain qualified.'),dict(parent='3451',citation='CDIAL[3451.1]',evidence='Full kṛṣṇa colour subsection 1 explicitly gives Kalasha kriẓṇa/krīṇḍa, Bshk kiṣin, Torwali kəṣən, Palula kiṣiṇu and Shina Kohistani kiṇŭ black, as well as Prakrit kasiṇa/kiṇha. These support the selected Dardic kṛṣṇa/kīṣan/kino colour responses. Survey rhotic retention/loss, vowel quality, sibilant notation, nasalization and inflection remain unchanged and their local phonetic history is qualified. This is colour-family membership, not a link to the proper-name subsection; uncertain cross-IA transmission remains explicit.')]
white={'panar','panār','paner','pānḍaro','pānar','panāro','paṇālo'}
black={'kirīśna','krīẓna','krīnḍā','kiṣinõ','kīṣan','kīṣān','kiśīn','kīśõ','kiśū̃','kiśo','kīśo','kīṣū̃','kino'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Gloss']=='white' and r['Form'] in white else 1 if r['Gloss']=='black' and r['Form'] in black else None
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare255.py').read_text());print('accepted',len(acc));print([(x['record']['Language_ID'],x['record']['Form'],x['parent']) for x in acc])
