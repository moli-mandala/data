import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'ninth-rules.json').read_text());cs=json.loads((P/'ninth-candidates.json').read_text());acc=[];held=[]
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];reason=None;rns=r['Language_ID']=='Rana' and 'Tharu-RNS-' in r['Tags']
 if len(x['families'])!=1:reason='Competing candidate parents need exact subsection review.'
 elif not rns and re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source uncertainty requires distinguishing lexical reading from locality attribution.'
 if reason:held.append(dict(**x,reason=reason,passNumber=9));continue
 ev=q['evidence']
 if rns:ev+=' RNS uncertainty concerns locality mapping only; the source audit establishes the lexical reading.'
 acc.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=ev))
(P/'ninth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','ninth-decisions.json').replace('sixth','ninth');(P/'ninth_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
