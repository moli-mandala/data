import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass90';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='7150',citation='CDIAL[7150];CDIAL[9661]',evidence='The complete younger-brother expression contains small/young + brother. CDIAL 7150 explicitly gives Lahnda/Punjabi nikkā small/young; CDIAL 9661 gives Lahnda bhrā, Punjabi bharā and regional bhāi brother. Two ordered component-family edges preserve the full nika/niko pra/bhāi expression. Initial p and breathy articulation in pra̤ are retained as regional qualifications; this is not a claim of a single inherited Sanskrit compound or a settled local loan route.'),dict(parent='7150',citation='CDIAL[7150];CDIAL[9349]',evidence='The complete younger-sister expression contains small/young + sister. CDIAL 7150 gives regional nikkā small/young with feminine nikī; CDIAL 9349 gives Lahnda bhēṇ, Punjabi bhaiṇ and regional sister forms. Two ordered component-family edges preserve niki beiṇ/pen/pe̤n/pǣṇ, including the source nasal and breathy articulation and p/bh difference as regional qualifications. No single inherited Sanskrit compound or resolved local transmission route is asserted.')]
bs={'nika pra','nika pra̤','niko bhāi'};ss={'niki pǣṇ','niki beiṇ','niki pen','niki pe̤n','niki pe̤ṇ'};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=0 if f in bs and g=='younger brother' else 1 if f in ss and g=='younger sister' else None
 if f=='niki bebu' and g=='younger sister':held.append(dict(record=r,families=[],reason='The initial niki is the small/young adjective, but bebu needs separate nursery-word/donor analysis. The reviewed bhaginī entry does not establish this second component, so the whole phrase is not yet linked.',passNumber=90))
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],components=['7150','9661' if i==0 else '9349'],family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass90_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print({'records':len(acc),'rows':len(acc)*2,'held':len(held)})
