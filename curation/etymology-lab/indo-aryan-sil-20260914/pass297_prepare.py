import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass297';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='12918',citation='CDIAL[12918]',evidence='Full saṃdhyā gives Pali sañjhā, regional sā̃jh/sā̃j, Kumaoni sā̃s and Assamese xā̃z evening. These support conservative sandʰyā/səndʰya and eastern śondha/śundu/śenda, alongside sãdz/sãsi/sãsa/sonz and Kullu son(d)z/ʃeːnəz. Preserve source vowel quality, nasalization and sibilant/consonant sequences; local reductions and conservative or cross-IA transmission remain qualified. Extra final segments and loss of the medial consonant without a diagnostic comparator are excluded.'),dict(parent='11625',citation='CDIAL[11625]',evidence='Full vikāla explicitly means evening and gives regional viāla/byāl, Assamese biyali and Hindi bīaṛ evening with its retroflex liquid marked uncertain. These support Hajong bikal and provisionally Danuwar bihar/biharai̯, Majhi behera and Bote bʸara/bʸiaro. Preserve source consonants and vowels; precise local h/r history and cross-IA transmission remain qualified, not a new regular sound law. Longer biharoko is not automatically stripped.')]
sets=[('evening',{'sandʰyā','səndʰya','śondha','ʃondha','śundu','śenda','sãdz','sãsi','sãsa','sonz','son(d)z','ʃeːnəz'}),('evening',{'bikal','bihar','biharai̯','behera','bʸara','bʸiaro'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare297.py').read_text());print('accepted',len(acc))
