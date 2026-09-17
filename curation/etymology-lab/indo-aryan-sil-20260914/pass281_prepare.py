import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass281';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9758',citation='CDIAL[9758,1]',evidence='Full matsya gives Prakrit maccha, Oriya mācha, regional māch/mācho, Marathi māsā and fish forms in -ī. The western matsha/matsho/māṭsi responses and Oriya ma.co fit this base family, preserving spacing, aspiration and affricate spelling without inventing a distinct etymon. Cross-IA transmission is qualified.'),dict(parent='9758-3',citation='CDIAL[9758]',evidence='The full matsya article explicitly separates the l-extension, giving Prakrit maścalī, West Pahari machlī, Awadhi macharī, Hindi machlī and Marathi/Konkani māsḷī. These support məcri/mačhri, mācəli/mācaḷ/mātsalu/macalu and Kullu matʃəɭi. Use the existing l-extension node 9758-3, not the separate ll-extension cited for Gujarati. Local vowel and liquid developments and cross-IA transmission remain qualified.'),dict(parent='9758-2',citation='CDIAL[9758,2]',evidence='The full addendum to matsiya section 2 explicitly gives Kotgarhi/Koci máċċhi fish, supporting Kullu moːtʃi/motʃi with source vowels and aspiration retained and the precise local sound history qualified.')]
sets=[('fish',{'ma.co','matsh a','matsha','māṭsi','matsho'}),('fish',{'məcri','mācaḷ','mācəli','mātsalu','matʃəɭi','macalu','mačhri'}),('fish',{'ˈmoːtʃi','motʃi'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare281.py').read_text());print('accepted',len(acc))
