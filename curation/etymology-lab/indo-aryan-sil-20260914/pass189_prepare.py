import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass189';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='f_gjvdfsrfn3muy',citation='oped2026[entry 17180]',evidence='The archived full OPED entry explicitly gives Pashto ǰolā́ both weaver and spider, including a spider-web example. Simple yolā/yolo/yūlā spider responses match the regional y-initial adaptation independently attested in the Ushojo survey for BOTH weaver and spider. Linked provisionally as a regional borrowing with the existing Pashto lexical donor; direct Pashto versus mediation through neighboring languages remains unresolved. Vowel length/quality and final vowel are preserved. This does not assert native descent from CDIAL jāla merely because that word denotes a web. Extended žālāū/žolāū and mixed elicitation responses remain pending.')]
sets=[{'yolā','yolo','yūlā'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='spider':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='borrowed',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
