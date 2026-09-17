import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass95'
assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='7150',citation='CDIAL[7150]',evidence='CDIAL 7150 niktá documents replacement by MIA *nikka and explicitly gives Lahnda/Punjabi nikkā small, young, alongside Awan nik shortness. The bare regional nikā/nika masculine and niki feminine responses are analysed as substantivised young one, hence child/boy/girl. Source single-k spelling is retained; local Indo-Aryan transmission remains open. This does not assert that the Sanskrit adjective itself meant child.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 if r['Language_ID'] not in {'poth','Goj'} or r['Form'] not in {'nika','nikā','niki'} or r['Gloss'] not in {'child','boy','girl','daughter','(own) child'}:continue
 q=rules[0];e=q['evidence']
 if r['Gloss']=='daughter':e+=' The daughter response is a context-specific kin use of the feminine young one; this narrower semantic extension is flagged for audit, rather than attributed as a dictionary gloss.'
 acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=e+' Exact response: '+r['Form']+' '+r['Gloss']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'-primary-articles.json')).write_text(json.dumps({'7150':json.loads((P/'pass89-primary-articles.json').read_text())['7150']},ensure_ascii=False,indent=1)+'\n')
(P/'pass95_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc)})
