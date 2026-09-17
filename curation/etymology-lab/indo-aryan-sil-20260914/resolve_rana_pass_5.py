import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'fifth-rules.json').read_text());cs=json.loads((P/'resolve-rana-candidates-5.json').read_text());accepted=[];held=[]
dardic={'Kohistani','Shinaic','Chitrali','Kunar','Pashai'}
allowed={1:{'Kal','Kho','Phal'},3:{'Tor'},4:{'Bshk','Gaw','Kal','Kho','Tor'},7:{'Bshk','Tor','Mai','Chil','Phal','Gowro'},8:{'Kho','Tor'},9:{'Bshk','Tor','Mai','Phal'},10:{'Bshk','Kho'},11:{'Bshk','Tor','Mai','Kal','Kho'},12:{'Bshk','Tor','Mai'}}
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];l=r['Language_ID'];w=r['Form'];cl=r['clade'];reason=None
 if False: reason=None  # Locality-only flag resolved below.
 elif cl in dardic and l not in allowed.get(i,set()):reason='The local Dardic form is not established by the selected article’s named comparanda; inspect its paradigm or local transmission before saving.'
 elif i==4 and l in {'Noiri','Vasavi'}:reason='The bare ai first-person form requires local pronoun history; the Dardic ai comparator cannot establish a Bhili analysis.'
 elif i==7 and w.startswith(('tʰ','th')) and l!='Bshk':reason='Short thā-type you form does not distinguish a remodeled plural from singular tvam without the local paradigm.'
 elif i==9 and l=='Bshk' and 's' not in w:reason='Gawri reduced yūn contrasts with the article’s yūsun; verify local shortening versus contact with the Torwali/Indus Kohistani form.'
 elif i==11 and (l=='Kho' or w=='ūg'):reason='Water form has a different final velar from the cited local comparator (Khowar uγ, Gawri ū); inspect local fricative voicing or dialect history.'
 if reason:held.append(dict(**x,reason=reason,passNumber=5));continue
 accepted.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=q['evidence']))
(P/'resolve-rana-decisions-5.json').write_text(json.dumps(dict(accepted=accepted,held=held),ensure_ascii=False,indent=1));print('accepted',len(accepted),'held',len(held))
