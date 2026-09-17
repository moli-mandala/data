import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass183';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='5731',citation='CDIAL[5731.1]',evidence='Full CDIAL tala subsection 1 explicitly gives Lahnda/Punjabi talī palm and West Pahari hattei tali, plural telī, palm. Simple Goj telī, Dogri tɘli and Pothwari teli match this feminine palm form with vowel notation preserved. Geminate *talla subsection 2 and shortened hastatala forms are distinguished; local IA transmission is unresolved.'),
 dict(parent='14024',components=['14024','5731'],citation='CDIAL[14024];CDIAL[5731]',evidence='Analyse this surface expression as hand plus surface/palm in that order: CDIAL hasta supplies eastern hāt/hāta hand and tala explicitly supplies Bengali tal/talā and Oriya taḷa surface/palm, originally in compounds with hand. Bengali hater/hatɛr is the genitive hand element; Bishnupriya ator/atol is interpreted as its h-less regional counterpart, with r/l notation qualified. The final tala/talu/tara identifies the surface element, with lateral/rhotic variation retained. Both lexical components are saved, not an invented inherited compound node; local IA transmission is unresolved.'),
 dict(parent='5731',components=['5731','14024'],citation='CDIAL[5731];CDIAL[14024]',evidence='Oriya toḷohato palm is analysed transparently in its source order as surface/palm plus hand. Full CDIAL tala gives Oriya taḷa surface, sole, palm and hasta gives Oriya hāta hand. Save both lexical components in order, preserving vowels and the compound boundary interpretation; no new ancient compound is reconstructed and local IA transmission is unresolved.')]
sets=[{'telī','tɘli','teli'},{'hater tola','hatɛr talu','atortalu','atortara','atoltala'},{'toḷohato'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='palm':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))
