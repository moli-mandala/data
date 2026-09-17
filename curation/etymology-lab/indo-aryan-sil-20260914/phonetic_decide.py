import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'};acc=[];held=[];qs=[];ix={}
extra={
'1670':'CDIAL 1670 ujjvala explicitly gives Maithili ujjar/ujar “white” and notes Bihar-to-Nepali transmission; the survey forms fit this family.',
'5086':'CDIAL 5086.1 jaṭā explicitly gives Bihari jar “root” and Nepali jari/jaro; the short-vowel target forms select this base branch.',
'6459':'CDIAL 6459 *duvāra explicitly gives Western Pahari duār and Nepali duwār “door”, including the retained vowel between d and v.',
'12918':'CDIAL 12918 saṃdhyā explicitly gives Hindi sā̃j(h), Bhojpuri sā̃jh and Prakrit saṃjhā “evening”.',
'5994-3':'CDIAL 5994.3 trīṇi gives Hindi tīn and Nepali tin; the Kului dental articulation mark does not select a different numeral stem.'}
for x in json.loads((P/'phonetic-expansion-candidates.json').read_text()):
 r=raw[x['record']['ID']]
 if r['ID'] in current:continue
 reason=None
 if len(x['parents'])!=1:reason='Multiple reviewed roots coincide under the broader phonetic discovery key; inspect the alternatives.'
 k=x['parents'][0];c=x['comparanda'][k][0]
 if k not in ix:ix[k]=len(qs);qs.append(dict(parent=k,citation=c['citation'],evidence=extra.get(k,c['evidence'])))
 i=ix[k]
 if k=='6333' and r['Gloss']=='sun':reason='Sun-only response needs a same-locality day comparator before adopting the divasa day family.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires local review.'
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=24));continue
 ev=extra.get(k,c['evidence'])+' Comparative check: '+c['record']['Language_ID']+' '+c['record']['Form']+' “'+c['record']['Gloss']+'” ('+c['record']['ID']+'). The exact survey spelling and phonetic marks remain untouched. Stress, vowel articulation, release marks and source palatal/sibilant notation were reviewed as local details; the full form and sense support the same family. The link does not settle uncertain transfer within Indo-Aryan.'
 acc.append(dict(record=r,family=i,parent=k,citation=c['citation'],evidence=ev))
(P/'phonetic-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'phonetic-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'phonetic_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','phonetic-decisions.json').replace('sixth','phonetic'))
print('accepted',len(acc),'held',len(held))
