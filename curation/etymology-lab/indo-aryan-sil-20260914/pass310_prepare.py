import json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass310';assert not (P/(stem+'-decisions.json')).exists()
d=json.loads((P/'pass310-pending-review.json').read_text());assert d['status']=='pending'
rules=[dict(parent='10924',citation='CDIAL[10924,1]',evidence='User explicitly selects laḍikka for this reviewed child group on 2026-09-15. Turner gives Punjabi laṛkā and Awadhi/Bhojpuri larikā in this branch, while allowing several other regional forms to derive alternatively from laḍḍikka with shortening. Retain that published alternative and qualified regional transmission; the selected branch is an editorial choice, not proof that the competing reconstruction is excluded.'),dict(parent='10247',citation='CDIAL[10247]',evidence='User explicitly selects mūrdhan for this reviewed head group on 2026-09-15. Turner records regional mur/muṇḍ head comparanda but explicitly allows derivation from or crossing with muṇḍa shaven for many unaspirated forms. Retain this possible contamination or alternative derivation and local phonetic/transmission uncertainty.')]
acc=[]
for i,g in enumerate(d['groups']):
 for x in g:
  q=rules[i];acc.append(dict(record=x['record'],family=i,kind='reflex',**{**q,'evidence':q['evidence']+' Prior hold, now resolved by user choice: '+x['reason']}))
assert len(acc)==115
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare310.py').read_text());print('accepted',len(acc))
