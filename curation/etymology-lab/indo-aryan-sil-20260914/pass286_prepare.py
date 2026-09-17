import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass286';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='1921',citation='CDIAL[1921,1]',evidence='Full udaka explicitly gives Khowar uγ, Kalasha ūk, Bshk ū, Phal/Mai/Gau wī and Chil woy water. These support ūxh/ūkʰ, ūā, ve/veʔ/vī and provisionally Palula bī with a strengthened labial onset. Source vowels, glottalization, frication and aspiration remain intact. Both vī alternatives share this family. Local developments and cross-IA transmission are qualified; longer responses with unexplained second words are excluded.'),dict(parent='7552',citation='CDIAL[7552]',evidence='Full nīra gives Prakrit ṇīra water, Lahnda nīr water and Hindi nīrā water, supporting Mewari nīr as an IA family link. The article discusses a Dravidian origin for the Sanskrit word, but this does not prove a direct Dravidian donor for the Mewari response. Immediate cross-IA transmission is qualified.'),dict(parent='8082',citation='CDIAL[8082]',evidence='Full pānīya gives Marwari/Gujarati/Marathi pāṇī, regional pāni and reduced nasal-vowel forms such as Poguli pāĩ. These support Mewari phaṇi and Dungra Bhili pãʔĩ provisionally, preserving source aspiration/glottalization and qualifying their local history and cross-IA transmission.')]
sets=[('water',{'ūxh','ūkʰ','bī','ūā','veʔ','ve','vī / vī'}),('water',{'nīr'}),('water',{'phaṇi','pãʔĩ'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare286.py').read_text());print('accepted',len(acc))
