import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass269';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8827',citation='CDIAL[8827];CDIAL[8848]',evidence='Full prasava 8827 supplies Middle Indo-Aryan pasava. Crucially, the full prasūna article 8848 explicitly cites prasava masculine/neuter flower in the Mahābhārata, supplying the flower sense absent from the short 8827 heading. Hadauti pasab fits the pasava family with final-vowel loss and b/v correspondence; the source spelling and flower gloss remain intact. This is a provisional family assignment with local transmission and the article’s broader sū/śvi collision discussion qualified, not an etymology inferred from procreation alone. Puṣpa was inspected as a separate comparison.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='flower' or r['Language_ID']!='had':continue
 if r['Form']=='pasab':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']))
 elif r['Form'] in {'pasam','pasāmb'}:held.append(dict(record=r,families=[0],passNumber=269,reason='Prasava/pasava has direct flower-sense support in CDIAL 8848, but the nasal ending of Hadauti pasam/pasāmb is not explained by the inspected entries. Puṣpa and prasūna were also compared. Need a local lexical comparison or documented b/m variation before treating this as the same stem; uncertainty goes beyond transmission alone.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare269.py').read_text());print('accepted',len(acc),'held',len(held))
