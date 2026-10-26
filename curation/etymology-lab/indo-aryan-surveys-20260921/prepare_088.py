import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
a=[];h=[]
for r in rs:
 if r['ID'] in done:continue
 if r['Gloss']=='broken' and r['Form'] in ['ādhɔ','ādo','adʰo','ada','ādɔ','ādʰɔ']:
  a.append(dict(record=r,parent='644',kind='reflex',citation='CDIAL[644]',evidence='Turner ardha half explicitly records Marwari ādho and neighbouring Hindi ādhā, with western ad-/ādh- variants. These Malvi/Nimadi responses to broken are the half/incomplete adjective, a transparent semantic alternative to a break participle. The elicited gloss is preserved rather than silently changed to half.',family=0))
 elif r['Gloss']=='whole' and r['Form']=='samūcā':
  h.append(dict(record=r,reason='Whole samūcā matches CDIAL 13227 samuccaya, but Turner explicitly says retained m is unexplained and proposes an early borrowing as a possibility. No specific immediate donor or secure local transmission is established here.'))
 elif r['Gloss']=='broken' and r['Form'] in ['apūrṇ','adūrā','adūro']:
  h.append(dict(record=r,reason='Broken/incomplete: apūrṇ appears learned from apūrṇa (493); adūrā/adūro may represent a different half-full/incomplete formation. Do not force them into simple ardha half without checking the relevant history and donor route.'))
(P/'pass088-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held')
