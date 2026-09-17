import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass214';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8056',components=['8056','11366'],citation='CDIAL[8056.1];CDIAL[11366];CDIAL[8068]',evidence='The Bareli pai-vaṭ path expression is foot plus way, in that order. Full CDIAL pāda subsection 1 gives Gujarati/Marathi pāy foot and vartman gives Gujarati/Marathi vāṭ path. Preserve source pai and short-vowel notation. The reconstructed pādavartman 8068 specifically supplies Gujarati pāvaṭ a sloping path into a tank; this modern transparent expression is instead represented by ordered components. Turner notes that some feminine path words can also continue vartis 11363, whose full entry explicitly gives Lahnda and Sinhala forms; the saved vartman assignment follows the Gujarati/Marathi regional grouping without resolving the deeper overlap or local IA transmission.'),dict(parent='7766',components=['7766','11366'],citation='CDIAL[7766];CDIAL[11366];CDIAL[11363]',evidence='The Vasavi pag-vaṭ path expression is foot plus way, in that order. Full CDIAL padga gives Gujarati pag/pāg foot and vartman gives Gujarati/Marathi vāṭ path. Preserve source vowels and source hyphen. Turner leaves the single g/vowel of padga reflexes unexplained and allows some feminine path words also to continue vartis, explicitly represented by Lahnda/Sinhala in 11363; these qualifications remain. Save the transparent components with the western regional vartman grouping, without asserting an ancient compound or settled local IA transmission.')]
sets=[{'pai-vaṭ'},{'pag-vaṭ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='path':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))
