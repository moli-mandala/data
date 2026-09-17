import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass303';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='13910',citation='CDIAL[13910];platts[655];nirmaan2018mewari[333,350]',evidence='Full CDIAL svarga gives heaven. Platts p.655 explicitly derives Hindi sarg/sarag/surag heaven, firmament, sky from Prakrit sarago and Sanskrit svarga. The Mewari dictionary p.333 records sarag paradise/heaven and p.350 harag as its variant; the Halbi dictionary also independently attests sarag heavens/sky. These support regional sarg/sarag/sorog and neighbouring Bhil harag forms. Source vowels, final devoicing or aspiration and initial s/h/x variation are retained, with exact local sound history and cross-IA transmission qualified. Forms ending in an unexplained nasal are excluded.')]
sets=[('sky',{'śarak','sārakʰ','sarag','sərgə','sərku','śərəgā','sərək','sorog','sərəg','sereg','surekh','sarak','sorag','sorak','xorig','sarig','harag','hərəg','sarog','sʌrəg','sʌrəgə','səreg','horog'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare303.py').read_text());print('accepted',len(acc))
