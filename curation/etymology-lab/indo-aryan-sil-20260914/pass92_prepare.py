import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass92';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
spec=[('5071','CDIAL[5071]','CDIAL 5071 *chōṭṭa gives regional choṭā/choṭo small. The selected aspirated choṭa adjective specifies the younger sibling.'),('12732','CDIAL[12732]','CDIAL 12732 ślakṣṇa gives Gujarati nānũ, Oriya sāna small/youngest, and Marathi lahān small. The selected nāno/nānā/nāni, san and lāhān adjectives specify the younger sibling; source vowels and nasal notation are retained.'),('10896-5','CDIAL[10896, -kk extension]','CDIAL 10896 gives halkā/halukā through the -kk extension and metathesis of laghu, with light/small/young senses in the family. Canonical 10896-5 *laghukk- is selected for the halko/halako/halaki adjective specifying the younger sibling.')]
rules=[]
for p,c,e in spec:
 for j,parent in enumerate(['9661','9349']):rules.append(dict(parent=p,citation=c+';CDIAL['+parent+']',evidence=e+' The second component is '+('brother (CDIAL 9661 bhrātṛ, regional bhāi/bhāū/bhrā).' if j==0 else 'sister (CDIAL 9349 bhaginī, regional bahin/bahan/bon/bhaüṇī).')+' Two ordered etymological component links preserve the complete response; they do not assert one inherited Sanskrit compound or resolve internal Indo-Aryan transmission.'))
bs=[{'cʰoṭā bāy','cʰoṭā bʰāī','cʰoṭɔ bʰāy','cʰoṭobʰāū','tʃhoʈo bhai ̯'}, {'nānobʰāī','lāhānbāu','san bai','san bʰai','nāno bɦāi','nano bhai','nānā bɦāy','nānobhāi'}, {'halakobʰāī '}]
ss=[{'cʰoṭībahaṇ','choto bon','tʃhoʈo bon','choṭibeh̰an','choṭibahan'}, {'nāni bohuṇ','nani bun','nāni bāiṇ','san boini','san bouni','san bʰoini','san bʰoni'}, {'halki bahin','halaki bɛn'}]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for k in range(3):
  j=0 if r['Gloss']=='younger brother' and r['Form'] in bs[k] else 1 if r['Gloss']=='younger sister' and r['Form'] in ss[k] else None
  if j is None:continue
  i=2*k+j;q=rules[i];acc.append(dict(record=r,parent=q['parent'],components=[q['parent'],'9661' if j==0 else '9349'],family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
a=json.loads((P/(stem+'-primary-articles.json')).read_text());old=json.loads((P/'pass90-primary-articles.json').read_text())
for k in ['9661','9349']:a[k]=old[k]
(P/(stem+'-primary-articles.json')).write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass92_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print({'records':len(acc),'rows':len(acc)*2})
