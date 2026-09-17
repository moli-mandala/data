import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'sixth-rules.json').read_text());cs=json.loads((P/'sixth-candidates.json').read_text());accepted=[];held=[]
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];l=r['Language_ID'];cl=r['clade'];w=r['Form'];reason=None;parent=q['parent'];kind='reflex';ev=q['evidence']
 rns=l=='Rana' and 'Tharu-RNS-' in r['Tags']
 if not rns and re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source/dialect uncertainty marker: establish whether it qualifies the locality or lexical reading.'
 elif cl in {'Kohistani','Shinaic','Chitrali','Kunar','Pashai'} and not (i==5 and l=='Kal'):reason='The exact Dardic reflex needs a local comparison; Khowar sutūr differs from CDIAL šutur and its discussed sibilant history.'
 elif i==2 and l not in {'N','Dotyali','jaun','kul'}:reason='The rāmro good-family is directly attested in Nepali; determine borrowing versus local inheritance in this other Nepal-area language.'
 elif i==6 and (cl in {'Bihari','W. Hindi','E. Hindi','Rajasthani'} or l in {'Goj','mewari_basad','kaithal'}):reason='CDIAL 8857 marks several Central patthar/pathar forms as Punjabi loans; establish immediate transmission rather than saving an inherited link.'
 elif i==7:
  if l=='H':parent='f_uci5okj4woxdw';kind='borrowed';ev='CDIAL 142 explicitly marks standard Hindi acchā as borrowed from Punjabi acchā. This link points to the existing Punjabi attestation f_uci5okj4woxdw, not directly to Sanskrit accha.'
  elif l in {'P','awan','poth'}:pass
  elif l=='Nimadi' and w=='ācho':pass
  else:reason='CDIAL 142 distinguishes inherited western āch- forms from Punjabi > Hindi acchā; the survey lect’s immediate borrowing route is not established.'
 if reason:held.append(dict(**x,reason=reason,passNumber=6));continue
 if rns:ev+=' The RNS uncertainty is the documented Sisaikhara/Sisana site mapping, not a lexical reading uncertainty; source tags are retained.'
 accepted.append(dict(record=r,family=i,parent=parent,citation=q['citation'],evidence=ev,kind=kind))
(P/'sixth-decisions.json').write_text(json.dumps(dict(accepted=accepted,held=held),ensure_ascii=False,indent=1));print('accepted',len(accepted),'held',len(held),'kinds',{k:sum(x['kind']==k for x in accepted) for k in {x['kind'] for x in accepted}})
