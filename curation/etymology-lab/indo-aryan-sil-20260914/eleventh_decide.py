import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'eleventh-rules.json').read_text());cs=json.loads((P/'eleventh-candidates.json').read_text());acc=[];held=[]
qs.append(dict(qs[13],parent='13519-2',citation='CDIAL[13519.2]'))
(P/'eleventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];reason=None;rns=r['Language_ID']=='Rana' and 'Tharu-RNS-' in r['Tags']
 if i==13:x['families']=[13,15];reason='CDIAL explicitly leaves suvarṇa versus sauvarṇa unresolved for these gold forms; the two existing parent nodes require an exact-branch decision, not merely a borrowing-path decision.'
 elif i==14 and r['Language_ID'] not in {'Bshk','Tor'}:reason='Retained-k nail form may continue strengthened *nakkha or conservative/learned nakha; the exact parent cannot be settled by the generic nail match.'
 elif not rns and re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source uncertainty requires distinguishing lexical reading from locality attribution.'
 if reason:held.append(dict(**x,reason=reason,passNumber=11));continue
 ev=q['evidence']
 if i==14:ev='CDIAL 6914.2 explicitly names Bshk. nakh and Torwali nōkh, selecting the strengthened *nakkha branch for these survey fingernail forms. Source vowel quantity/nasalization is retained.'
 if rns:ev+=' RNS uncertainty concerns locality mapping only; the source audit establishes the lexical reading.'
 acc.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=ev))
(P/'eleventh-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','eleventh-decisions.json').replace('sixth','eleventh');(P/'eleventh_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
