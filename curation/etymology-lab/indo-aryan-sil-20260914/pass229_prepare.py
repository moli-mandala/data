import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass229';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='1728',citation='CDIAL[1728]',evidence='Full CDIAL utkuṇa explicitly includes Prakrit okkaṇī louse, Bengali ukun/okan/okni/ikun and Oriya ukhan/ukhun/ukhen head-louse. Danuwar okʷon/okʷan/okʰon louse fits this family, preserving rounded/labialized k versus aspirated kh notation and vowels. The phonetic distinction is not silently normalized, and the local IA transmission route remains unresolved.'),dict(parent='9085',citation='CDIAL[9085]',evidence='Full CDIAL phutta explicitly gives Bshk phut fly/phit mosquito and Phalura phutto fly/phutti mosquito. The selected Phalura pūtī and regional Maiya/Ushojo phūtī/phūthī and Bhateri phūth mosquito forms match this lexical family, with loss or placement of aspiration, vowel length and final feminine-vowel variation retained as qualifications. Both fly and mosquito are documented within the family, so the survey mosquito sense is preserved; no local donor route or independent derivation from the abstract insignificant gloss is asserted. Mixed phīth pīlīl and other two-part responses are excluded.')]
sets=[{'okʷon','okʷan','okʰon'}, {'pūtī','pʰūtī','pʰūtʰī','pʰūtʰ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in ('louse','mosquito'):continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms and r['Gloss']==('louse' if i==0 else 'mosquito'):
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
