import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass139';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4070',citation='CDIAL[4070]',evidence='CDIAL 4070 gala throat/neck explicitly gives Bengali galā throat and cognate neck senses in neighbouring Indo-Aryan languages. Bengali gola/gɔla neck retain the survey rendering of the Bengali vowel; the throat/neck overlap is directly documented in the primary family. Neither spelling is silently normalized, and local transmission is unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in {'neck','throat'}:continue
 if r['Language_ID']=='B' and r['Form'] in {'gola','gɔla'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 else:
  reason='Bishnupriya nar neck: 7047 nāḍī has Hindi nār windpipe and Assamese neli throat, while 6936 naḍa and its derivatives also have gullet/windpipe senses. These establish plausible comparisons but not the specific stem or the outer-neck sense for this language; an attested regional nar comparison is needed.' if r['Form']=='nar' else 'Ushoji tūrūṛ throat: homepage discovery for turu with throat returned no lexical match. No supported etymon or secure segmentation established in this pass; retain the exact source form for further dictionary/source research.'
  held.append(dict(record=r,families=[],reason=reason,passNumber=139))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
