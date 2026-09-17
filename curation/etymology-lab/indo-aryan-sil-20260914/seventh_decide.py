import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'seventh-rules.json').read_text());cs=json.loads((P/'seventh-candidates.json').read_text());acc=[];held=[]
qs[4]['parent']='6624-2';qs[4]['citation']='CDIAL[6624, -ḍ- extension]'
(P/'seventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];reason=None
 rns=r['Language_ID']=='Rana' and 'Tharu-RNS-' in r['Tags']
 if not rns and re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source uncertainty requires distinguishing lexical reading from locality attribution.'
 elif i==2 and r['Language_ID']=='Phal':reason='Palula muro needs local vowel/paradigm evidence beyond the article’s mār- comparison.'
 elif i==3 and r['Form'] in {'dekʰle'}:reason='Final -le may be the light verb take rather than ordinary inflection; analyse the construction before a whole-form reflex link.'
 if reason:held.append(dict(**x,reason=reason,passNumber=7));continue
 ev=q['evidence']
 if rns:ev+=' RNS source uncertainty concerns locality mapping only; the source audit establishes the lexical reading.'
 acc.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=ev))
(P/'seventh-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','seventh-decisions.json').replace('sixth','seventh');(P/'seventh_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
