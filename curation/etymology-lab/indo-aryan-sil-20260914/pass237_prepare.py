import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass237';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11165',citation='CDIAL[11165]',evidence='Full CDIAL lohita explicitly gives Jaunsari loī blood, Kului lóu, Gujarati lohī and regional lūi/lō blood forms. The selected loy/ḷui/luhi/luhĩ/lũhi/lo(h)u/ḷo responses fit this blood family, preserving lateral retroflexion, nasalization, glide notation and optional h. Regional IA transmission remains unresolved; no source form or blood gloss is normalized.'),dict(parent='11165',citation='CDIAL[11165]',evidence='Full CDIAL lohita blood explicitly includes Oriya lohu/nohu and la(h)u/na(h)u, establishing an n-initial variant within the l-initial blood family. The selected western Bhil-area nuhi/noi/nuye/noye responses are provisionally grouped with this family through that documented l/n alternation and regional lohi/lui comparisons. This does not assert an Oriya loan or prove the local history of n; local transmission and vowel reduction remain qualified. Source spellings and blood glosses are unchanged.')]
sets=[{'loy','ḷui','luhi','luhĩ','lũhi','lo(h)u','ḷo'}, {'nuhi','noi','nuye','noye'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='blood':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare237.py').read_text());print('accepted',len(acc))
