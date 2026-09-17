import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass179';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='13720-2',citation='CDIAL[13720]',evidence='Full CDIAL stoka explicitly labels an extension in -ḍ-, with Hindi/Punjabi thoṛā, Gujarati thoṛũ and Marathi thoḍā a little/few; Jaunsari thoṛō and Kotgarhi thoṛɔ also mean short. The selected simple thuḍ-/thoḍ-/toḍ-/thuṛ- forms match that extension, retaining vowel and aspiration variation. The canonical extension node is 13720-2; the primary article does not number it subsection 2, so the citation is CDIAL[13720]. Local IA transmission is unresolved. Extra -k/-so compounds are excluded.'),
 dict(parent='13720-3',citation='CDIAL[13720]',evidence='Full CDIAL stoka separately labels its -l- extension and gives Lahnda thōlā a little/few. The thulo few responses match that lateral branch with retained vowel variation, rather than the rhotic/retroflex -ḍ- branch. The canonical node is 13720-3; the printed extension is unnumbered. Local IA transmission is unresolved.'),
 dict(parent='12732',citation='CDIAL[12732]',evidence='Full CDIAL ślakṣṇa gives Prakrit laṇha/saṇha small, Punjabi nannhā/nannā, Hindi nanhā and Gujarati nāhnũ/nānũ small. Simple survey nahna/nānu/nani short follow this diminutive-size family with ordinary ending variation and a small-to-short sense extension explicitly qualified. Extra -l/-k/-c forms are excluded. Local IA transmission is unresolved.')]
sets=[set('thuḍo thuḍa toḍo tuḍo thuṛo ṭʰoṛī tʰoḍa'.split()),{'thulo'},{'nahna','nānu','nani'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in {'few','short'}:continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
