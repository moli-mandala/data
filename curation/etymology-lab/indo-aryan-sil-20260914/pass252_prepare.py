import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass252';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5239',citation='CDIAL[5239]',evidence='Full CDIAL jīva explicitly gives Oriya jī life/mind/heart and, in the addendum, West Pahari jiu mind/heart with oblique jiwa/jiba; Romani ǰi heart/soul is also supplied. Bhatri jiu̯, Rathwi dziu, Nimadi ju and Vasavi jib heart fit this regional jīva family with source affricate, glide, b/v and vowel reduction preserved. Mewari jivaḍo has a regional extended comparison in Gujarati jivṛo life, with its -ḍo and local history qualified. Regional IA transmission remains unresolved. The heart sense is explicit in primary prose, not guessed solely from life; longer jiban-type or heartbeat compounds are excluded.')]
forms={'jiu̯','dziu','ju','jib','jivaḍo'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='heart' and r['Form'] in forms:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare252.py').read_text());print('accepted',len(acc))
