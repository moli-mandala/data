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
for x in json.loads((P/'semantic-expansion-candidates.json').read_text()):
 r=raw[x['record']['ID']]
 if r['ID'] in current:continue
 reason=None
 if len(x['parents'])!=1:reason='Multiple reviewed roots coincide under the broader semantic discovery key; inspect the alternatives.'
 k=x['parents'][0];c=x['comparanda'][k][0]
 if k not in ix:ix[k]=len(qs);qs.append(dict(parent=k,citation=c['citation'],evidence=extra.get(k,c['evidence'])))
 i=ix[k]
 if k=='5889' and 'pl' in r['Gloss']:reason='Plural use of a tu singular-family form needs local pronominal usage evidence.'
 if k=='6507-2' and 'lo' in r['Form']:reason='Final lo may be a take light verb; resolve segmentation before selecting an inherited verb link.'
 if k=='1351' and r['Language_ID'] not in {'Bshk','Gaw','Tor'}:reason='Plains aya mother requires distinguishing the nursery *āī family from āryikā.'
 if k=='6331':reason='Contracted di day can also reflect divasa in the primary discussion; the short form needs a local historical account.'
 if k=='6333' and r['Gloss']=='sun':reason='Sun-only response needs a same-locality day comparator before adopting the divasa day family.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires local review.'
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=25));continue
 ev=extra.get(k,c['evidence'])+' Comparative check: '+c['record']['Language_ID']+' '+c['record']['Form']+' “'+c['record']['Gloss']+'” ('+c['record']['ID']+'). The exact survey spelling and semantic marks remain untouched. Equivalent elicitation glosses and ordinary inflections were reviewed together; source person, number, honorific and tense labels remain in the original record. The full form and sense support this same family. The link does not settle uncertain transfer within Indo-Aryan.'
 acc.append(dict(record=r,family=i,parent=k,citation=c['citation'],evidence=ev))
(P/'semantic-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'semantic-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'semantic_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','semantic-decisions.json').replace('sixth','semantic'))
print('accepted',len(acc),'held',len(held))
