import json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass309';assert not (P/(stem+'-decisions.json')).exists()
acc=[];rules=[]
for n in [277,283,285,288]:
 for x in json.loads((P/f'pass{n}-decisions.json').read_text())['held']:
  r=x['record']
  if n==283 and r['Form'] not in {'bājro','kodro'}:continue
  parent=('3515' if n==277 or (n==283 and r['Form']=='kodro') else '9201' if n==283 else '9712' if n==285 else '3674')
  note=('User accepts the cereal-family match despite rice/barley versus millet semantics. Preserve the survey gloss and flag cereal identification or semantic extension as unresolved.' if n in [277,283] else 'User selects majjan rather than medya for this fat form. Preserve the published irregular phonology and possible crossing with the medas/medya family.' if n==285 else 'User accepts ksara for these reduced ash forms. Preserve uncertainty about liquid loss and the source glottalization; the alternative khak comparison is not selected.')
  ev=note+' Earlier audit: '+x['reason']+' This previous hold is resolved by explicit user editorial choice on 2026-09-15; its evidential qualifications remain.'
  q=dict(parent=parent,citation='CDIAL['+parent+']',evidence=ev);i=len(rules);rules.append(q);acc.append(dict(record=r,family=i,kind='reflex',**q))
held=[]
for x in json.loads((P/'pass301-decisions.json').read_text())['held']:
 x['reason']='User decision 2026-09-15: do not link this star form. Retain unlinked; no approval to assign śukra or śukla. Prior evidence: '+x['reason'];x['passNumber']=309;held.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare309.py').read_text());print('accepted',len(acc),'deliberately unlinked',len(held))
