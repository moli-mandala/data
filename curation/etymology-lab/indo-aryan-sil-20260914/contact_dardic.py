"""Individual family review of the earlier blanket Dardic restriction."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;acc=[]
# Family matches inspected jointly; keep star/cow/new where branch choice remains open.
parents={'11165','10223','1268','2712','5958','7067','7655','8330','6141','9758','12583','9209','9349','8283','941','10158','4368','6849','13271','5679','10539'}
for x in json.loads((P/'audit-records.json').read_text()):
 generic=x['reason'].startswith('The general Indo-Aryan')
 ten=x['reason'].startswith('The s-final ten-form')
 if not (generic and len(x['candidates'])==1 and x['candidates'][0]['parent'] in parents or ten):continue
 q=x['candidates'][0];ev=q['evidence'];ev=' '.join(ev) if isinstance(ev,list) else ev
 ev+=' The form and meaning identify this comparative family. Under the user’s explicit cross-Indo-Aryan policy, the link is saved provisionally despite uncertain inheritance versus regional borrowing; the reflex relation does not assert an established immediate donor or uninterrupted inheritance. Prior transmission hold: '+x['reason']
 cite=q['citation'];cite=';'.join(cite) if isinstance(cite,list) else cite
 acc.append(dict(record=x['record'],parent=q['parent'],citation=cite,evidence=ev,family=q['parent'],transmissionUncertain=True,priorHold=x['reason']))
(P/'contact-dardic-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','contact-dardic-decisions.json').replace('sixth','contact-dardic');(P/'contact_dardic_save.py').write_text(s)
print(len(acc))
