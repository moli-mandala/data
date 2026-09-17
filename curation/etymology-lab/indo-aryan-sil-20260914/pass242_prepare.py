import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass242';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6152',citation='CDIAL[6152]',evidence='Full CDIAL danta explicitly gives regional dā̃t tooth across Nepali, Bihari, Hindi and western IA, alongside dand and the proposed Nepali dā̃d stage. The selected dãṭ/ḍãṭ/ḍaṭ/ḍãt/dath and Dotyali dā̃dā tooth responses fit this family with source dental/retroflex notation, nasalization, aspiration spelling and final vowel retained. Local IA transmission is unresolved. The separate non-nasal Dotyali dāḍā and longer compound/extended tooth forms are excluded for separate review.'),dict(parent='7031',citation='CDIAL[7031.1]',evidence='Full CDIAL nasta subsection 1 explicitly gives Torwali natkōl nose and Maiyan nathūr nose. Torwali netkel and Mai nasūr/nāsūr are provisionally grouped with those exact regional comparisons, preserving source vowels and s versus th notation. Turner marks the extension of natkōl and related forms with uncertainty; no independent origin of the whole extension or component segmentation is asserted. Regional transmission remains unresolved. This uses the nose-bearing branch, not the separate nastā nose-ring branch.')]
sets=[{'dãṭ','ḍãṭ','ḍaṭ','ḍãt','dath','dā̃dā'},{'netkel','nasūr','nāsūr'}];glosses=['tooth','nose']
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,forms in enumerate(sets):
  if r['Gloss']==glosses[i] and r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare242.py').read_text());print('accepted',len(acc))
