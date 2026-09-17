import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass126';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10158',citation='CDIAL[10158]',evidence='CDIAL 10158 mukha explicitly gives mouth/face, Prakrit muha, Lahnda/Punjabi mũ(h), Maiya mũ, Bengali mu and Oriya muhã; it also documents mui/mūī face forms. Selected mṳ̃/mhū̃/mũ./mũə/məh retain breathiness, nasalization, vowel and h placement, while western mvi/moi retain the glide/final-i shape as a regional qualification. These are short-form family links, not source-spelling corrections; local Indo-Aryan transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='face' and r['Form'] in {'mvi','moi','mṳ̃','mhū̃','mũ.','mũə','məh'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
