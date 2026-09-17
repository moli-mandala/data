import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass80';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4147-4',citation='CDIAL[4147, -ḍa extension]',evidence='CDIAL 4147 explicitly gives Garhwali gauṛī cow and Gujarati gāvṛī affectionate cow in its -ḍa extension. Jambu promotes that extension to the existing node 4147-4 *gāvaḍa-. Western gauḍi/gāvḍi/gavaḍi forms retain the cow stem plus this retroflex extension, with stop versus flap, the av/au sequence and final feminine i preserved. Exact intra-Indo-Aryan transmission remains open; this is more specific than assigning the unextended cow heading.')];acc=[];held=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=r['Form'];g=r['Gloss']
 if w in ['gauḍi','gāvḍi','gavaḍi','gāvaḍi'] and g=='cow':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+w+' is preserved.'))
 elif w in ['ay','āy'] and g=='today':held.append(dict(record=r,families=[],reason='The homepage finds a linked āž/āy response under adya, but CDIAL 242 itself gives local ajj/az forms rather than an unconditioned y. Establish the local affricate-to-glide development or source reading before extending that mixed-response comparison.',passNumber=80))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass80_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
