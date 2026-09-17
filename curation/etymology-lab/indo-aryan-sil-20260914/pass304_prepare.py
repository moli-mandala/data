import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass304';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='549',citation='CDIAL[549,1]',evidence='Full abhra section 1 explicitly records Mai azu, Chilisso and Gowro azo cloud/rain alongside Shina azu. These directly support the selected northern azo/aza/azu cloud responses; preserve survey sibilant and vowel variation. Cross-IA transmission and exact local sound developments remain qualified.'),dict(parent='549-2',citation='CDIAL[549,2]',evidence='Full abhra section 2 explicitly derives Gawri a(l)beno cloud from abhriya with a secondary suffix. Survey Gawri albena directly fits this comparative form, retaining its source final vowel and the published suffix qualification.'),dict(parent='549',citation='CDIAL[549,1]',evidence='Full abhra section 1 records Bengali abh/ab cloud and Gujarati/Marathi abh sky/clouds, with Old Marwari abhai in the sky. These support Hajong ap cloud with final devoicing and Mewari ab sky with aspiration absent in the source transcription. Regional transmission and exact phonetic history remain qualified.')]
sets=[('cloud',{'āžo','āẓo','āẓa','āžū'}),('cloud',{'albena'}),('cloud',{'ap'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if (r['Gloss']==g and r['Form'] in fs) or (i==2 and r['Gloss']=='sky' and r['Form']=='ab'):
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare304.py').read_text());print('accepted',len(acc))
