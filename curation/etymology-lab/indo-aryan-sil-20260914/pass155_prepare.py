import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass155';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='3674',citation='CDIAL[3674]',evidence='CDIAL 3674 kṣāra explicitly distinguishes widespread chār ashes from mostly khār alkali, including Jaunsari and western Pahari comparanda in the addendum. Selected car/char/tsar responses fit the ash family; aspiration, affricate/sibilant notation and final vowels remain qualified and unchanged. This does not equate every short ash word with kṣāra; local IA transmission remains unresolved.'),dict(parent='10555',citation='CDIAL[10555]',evidence='CDIAL 10555 *rakṣāpuṭaka explicitly supplies Gujarati rākhɔṛo/rākhɔṛī ashes or layers of ashes. The western Bhil rukhəḍ/rakhuḍ/rokhoḍ ash series matches that extended ash family, with r/ḍ notation, initial-vowel variation, deaspiration and endings qualified. It is kept distinct from plain rakṣā 10552 and nasal *rakṣākuṇḍaka 10553 ash-pit. No source spelling is corrected; local transmission is unresolved.'),dict(parent='10552',citation='CDIAL[10552]',evidence='CDIAL 10552 rakṣā ashes explicitly gives Sindhi rakha and western rākh. The short rak ash responses match this family with deaspiration qualified. Turner itself queries IA borrowing for Punjabi rākh; the link does not settle local transmission.')]
sets=[{'tʃʰaɾ','cār','cʰaᵒr','saur','sar','cʰar','car','char','caru','tʃʰaːr','tʃaːrə','tʃaːr','(t)saːr'}, {'rukhudu','rukhadu','rukhuḍu','rukhəḍo','rukhəṛu','rukhəḍā','rokhəḍo','rākḍo','rokhuḍu','rokhaḍā','rokhḍu','rekhaḍā','rakhuḍu','rəkhuḍo','rokhoḍo','rukuḍu','rukhaḍu'}, {'rak'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='ash':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
