import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass151';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10779',citation='CDIAL[10779]',evidence='The full addendum to CDIAL 10779 rudhyate explicitly derives Nepali rujhanu get wet here, preferring it to rīyate and citing western Pahari support. Dotyali rujeko/ružeko and Jaunsari ruji wet fit that regional wetting family; deaspiration, j/ž/y notation and participial endings remain qualified. This uses the documented wet sense, not an inference from the head gloss obstructed. Local transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='wet':continue
 if r['Form'] in {'ružeko','rujeⁱko','rujʸæko','ruji'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'sijya','sija','sijʸa'}:held.append(dict(record=r,families=[],reason='CDIAL 13393 sicyate explicitly gives L/P sijj- wet with analogical jj, while 13933 svidyati also gives L/P sijj- wet and Kangri sijjā damp; both cross-reference the competitor. These Pothwari sijya/sija forms need independent evidence to choose between those roots. Sikta 13388 does not resolve the voiced-affricate stem. This is competing etymology, not merely uncertain IA transmission.',passNumber=151))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc),'held',len(held))
