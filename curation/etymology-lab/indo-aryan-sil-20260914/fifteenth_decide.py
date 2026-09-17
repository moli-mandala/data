import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
known={}
for f in ['twelfth-rules.json','thirteenth-rules.json','fourteenth-rules.json']:
 for q in json.loads((P/f).read_text()):known[q['parent']]=q
skip={'1670','5086','12918','6459','11225','6507-2'}
qs=[];ix={};acc=[];held=[]
for x in json.loads((P/'comparative-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1:continue
 k=x['parents'][0]
 if k in skip or k not in known:continue
 r=x['record'];reason=None
 if k=='6663' and r['Language_ID'] in {'Chil','Tor','poth'}:reason='dar has competing dvara/dvāra branches or Persian origin; exact similarity to Jaunsari does not decide the donor or vowel history.'
 elif re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Review source-specific uncertainty before accepting.'
 if k not in ix:ix[k]=len(qs);qs.append(known[k])
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=15));continue
 comps=x['comparanda'][k];z=comps[0]['record']
 ev=known[k]['evidence']+' Joint comparison: '+r['Language_ID']+' '+r['Form']+' “'+r['Gloss']+'” matches reviewed '+z['Language_ID']+' '+z['Form']+' “'+z['Gloss']+'” ('+z['ID']+'). The shared family is provisionally linked under the user’s preference; inheritance versus intra-IA borrowing is not resolved by this formal comparison.'
 acc.append(dict(record=r,family=i,parent=k,citation=known[k]['citation'],evidence=ev))
(P/'fifteenth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'fifteenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'fifteenth_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','fifteenth-decisions.json').replace('sixth','fifteenth'));print(len(acc),len(held))
