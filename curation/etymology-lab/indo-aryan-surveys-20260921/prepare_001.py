"""Explicitly reviewed lexical groups; no similarity-driven acceptance."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
groups=[
dict(parent='9459',gloss='heavy',languages=['srk','awan','Goj'],forms=['bʰārā','bʰāro'],citation='CDIAL[9459]',evidence='Turner explicitly records Lahnda and Punjabi bhārā “heavy” under bhāra, alongside Kashmiri bāra. These northwestern a/o-final adjectives fit that attested adjective family; they are distinguished from i-final bhārī under bhārika (9465). The Gojri o-ending is the regional masculine ending; cross-Indo-Aryan transmission remains unspecified.'),
dict(parent='13845',gloss='broken',forms=['pʰūṭalṽ','pʰutal','putla','pʰutila','putila','puṭla','pʰuṭla','phūtlo','phūtɬa','pūṭlo','phūṭlo','phūṭlū','phūtelū','phūṭɬo','phūṭeḷ','pʰūtla','pʰuṭeḍo','pʰūṭna','phūṭinɔ','phūṭūnū','phūṭiyā','phuṭalõ','puṭṭio','puṭno','phuṭlo','phuṭlu','pʰut','puṭʰ','puṭ','pūṭ'],citation='CDIAL[13845]',evidence='Full *sphuṭyati gives Prakrit phuṭṭaï “bursts” and phuṭṭa “burst”, Nepali phuṭnu, Maithili phūṭab “be broken”, Bhojpuri phūṭal, Odia phuṭibā and Marathi phuṭṇẽ. The selected survey responses preserve the phuṭ-/puṭ- break stem with regional infinitive or participial endings; these inflections do not warrant derived edges. Deaspiration and the source dental/retroflex spellings are retained. Responses with a separate go-auxiliary, unexplained material, or a different slash-separated stem are excluded.'),
dict(parent='12548',gloss='dry',languages=['kaithal','Goj','Bhili','Bhilali'],forms=['sukʰā','suqqā','hūka','hukhā'],citation='CDIAL[12548]',evidence='Full śuṣka lists Prakrit sukkha/sukka, Lahnda and Punjabi sukkā and Hindi sūkhā “dry”. The selected Haryanvi form directly matches sūkhā; Gojri suqqā retains the geminate velar with the survey uvular spelling, and neighbouring Bhil hūka/hukhā show the regional s-to-h development. Final vowels are adjectival inflection. The assignment identifies the lexical family without fixing the regional transmission route.'),
dict(parent='12548-2',gloss='dry',languages=['Tor'],forms=['śūgel'],citation='CDIAL[12548, ext. -ll-]',evidence='Turner explicitly places Torwali šugil “dry” in the -ll- extension of śuṣka, beside Odia sukhilā. Survey śūgel has the same š-g-l skeleton and meaning, with vowel quality/length preserved; the specific extension node 12548-2 is selected rather than the bare śuṣka node.'),
]
acc=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['Gloss']==g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   acc.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(acc)==len({x['record']['ID'] for x in acc})
(P/'pass001-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass001-decisions.json').write_text(json.dumps({'accepted':acc,'held':[]},ensure_ascii=False,indent=1)+'\n')
print('Selected',len(acc),'records',[(g['parent'],sum(x['family']==i for x in acc)) for i,g in enumerate(groups)])
