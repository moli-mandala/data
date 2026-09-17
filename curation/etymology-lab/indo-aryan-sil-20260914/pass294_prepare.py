import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass294';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11742',citation='CDIAL[11742]',evidence='Full vidyut gives Palula bijji/biji, Gujarati vīj, regional bijj/vījj and Konkani ij lightning. These support Palula bīyī with weakened medial affricate provisionally, western bidz/vidz/βij/βidz and Dungra uije with reduced initial labial. Preserve the source affricate/fricative/vowel notation and qualify precise local history and cross-IA transmission. Longer added endings are not stripped.'),dict(parent='11745',citation='CDIAL[11745]',evidence='Full vidyullatā gives Prakrit vijjullayā/vijjuliā/vijjulī, regional bijulī/bijlī and Gujarati vijḷī lightning. These support βidzaḷi/vižəḷi/vidzḷi/bidzali/bijale/bidzale and provisionally ijiḷi with initial loss. Preserve lateral quality and final vowels; local changes and cross-IA transmission are qualified. The longer l-containing family is not collapsed into bare vidyut.')]
sets=[('lightning',{'bīyī','βij','βidz','bidz','vidz','uije'}),('lightning',{'βidzaḷi','vižəḷi','ijiḷi','vidzḷi','bijale','bidzale','bidzali'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare294.py').read_text());print('accepted',len(acc))
