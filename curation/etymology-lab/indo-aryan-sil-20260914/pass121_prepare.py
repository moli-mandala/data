import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass121';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4445',citation='CDIAL[4445]',evidence='CDIAL 4445 gharma gives Kumauni/Nepali ghām sunshine and Bhojpuri/Awadhi ghām sunshine, heat of the sun; the article also documents the sun sense elsewhere. Nepal-area gʰam/ɡʰam sun fits this regional noun with the metonymic sunshine-to-sun shift explicitly retained. Source length is preserved and local Indo-Aryan transmission remains unresolved.'),dict(parent='13574-2',citation='CDIAL[13574.2];platts1884[697–698]',evidence='Platts p. 697 explicitly derives Hindi sūraj/sūrj from Sanskrit sūryaḥ; p. 698 gives sūrya sun. CDIAL 13574.2 is the sūrya subsection. Selected surj/surz/surc and sury forms retain source voicing, aspiration, vowel and palatal notation. This chooses the ry-family with primary support, not the undifferentiated sūra head; learned or regional contact influence and the exact immediate Indo-Aryan route remain unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
ss={'sūryɛ','sūrac','suraź','surza','surdza','surjʰa','surja','suroj','suržʸæ','surjʰə','surdžo','surujʰ','surye','surje','sury'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='sun':continue
 i=0 if r['Form'] in {'gʰam','ɡʰam'} else 1 if r['Form'] in ss else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['13574']=json.loads((P/'pass119-primary-articles.json').read_text())['13574'];a['platts']=json.loads((P/'pass121-platts-research.json').read_text());f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
