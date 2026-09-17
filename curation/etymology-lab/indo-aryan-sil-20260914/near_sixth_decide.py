import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'3164':'CDIAL 3164 kim expressly gives Kalasha kīa, Ku./Hindi kyā and related what interrogatives. These survey kya/kea forms match that interrogative family.',
'10539':'CDIAL 10539 rakta explicitly covers both red and blood, including Bshk rat, K. rath and L. rattā. The source red/blood forms fit that semantic family; prefixed arattā needs separate ārakta comparison.',
'7150':'CDIAL 7150 nikta gives nīk/nīkā good through the MIA replacement *nikka and notes Persian nēk influence on nekā. I-vowel good forms fit this family; bare nek remains a potential Persian loan.',
'9661':'CDIAL 9661 bhrātṛ gives bhāi/bhāī/bhāua brother and ordinary regional forms. The survey bhaiya/bhaiwa and phāī fit that family with inflection/address endings and initial aspiration preserved.',
'11745':'CDIAL 11745 vidyullatā gives bijlī/bijulī/bījalī lightning. The survey y-bearing forms identify that same compound family; an n-bearing form needs independent explanation.',
'8249':'CDIAL 8249 puccha gives puch/pūchī tail and explicitly separates *puñcha in the addendum for nasalized forms. Unnasalized putʃhi fits the first branch; nasalized tails are retained for exact subsection resolution.',
'11348':'CDIAL 11348 *varta gives Gawri wāṭ, Kalasha bat and northern baṭ/baṭh stone. The b/v forms fit this family; initial g requires a distinct stone-word comparison.',
'6658-2':'CDIAL 6658.2 duvādaśa expressly gives Torwali duāš and neighbouring duwāleš twelve. The survey doāś/dūvāś forms fit that branch; bare Kalasha dūā needs numeral-composition review.',
'4368':'CDIAL 4368 grāma expressly gives Bshk lām and L./P. girā̃/grā̃ village. These forms fit the village family; the final p in lāmp needs a local transcription/phonetic account.',
'2871':'CDIAL 2871 karpaṭa gives kapṛā/kapṛo/kāpar cloth and clothing. These simple kapra responses fit that family; the deeper reconstruction remains qualified in Turner’s addendum.',
'12583':'CDIAL 12583 śṛṅga explicitly gives Bshk ṣīṅ, northern ṣiṅga and Hindi sī̃g horn. These singular/plural variants match the horn family with final nasal and glottal details preserved.',
'4225':'CDIAL 4225.2 *gūttha expressly contains Bshk gūt excrement, distinct from the first-branch gū forms. The retained-t survey forms are therefore linked to section 2.',
'10875-2':'CDIAL 10875.2 *lakkuṭa gives lakṛī/lakkaṛ wood and firewood, distinct from the first branch lauṛa. These retained-k forms select the second branch.',
'6943':'CDIAL 6943 nadī gives the river family, but these n/l/v alternations or a masculine nad response need individual sound and nad/nadī analysis before a link.',
'10990':'CDIAL 10990 laśuna gives lasun/lasaṇ garlic in its first branch and rasuna in its second. The simple l-initial forms fit the first branch; larasun has extra rhotic morphology needing review.',
'5958':'CDIAL 5958 taila explicitly gives Kalasha teu and regional tel/tilla oil. These teo/tyal/tiḷ responses fit the oil noun; source meanings distinguish it from tila sesame.',
'9883':'CDIAL 9883 markaṭa spider gives Nepali mākuro and regional makṛā/makarā. These spider forms fit that family, distinct from the similar monkey head.',
'12918':'CDIAL 12918 saṃdhyā gives sañjhā/sā̃j/sā̃z evening. These affricate/fricative variants identify the evening family.',
'6227':'CDIAL 6227 daśa gives Palula daš and regional das ten. These survey retroflex/dental and sibilant variants fit the numeral family.',
'13415':'CDIAL 13415 sindhu gives Khowar sin and L. sinnh/sĩdh river. The s-initial river forms fit; Bshk nīn needs comparison with nadī rather than an assumed s-loss.',
'6333':'CDIAL 6333 divasa explicitly gives Gawri dōs, Savi dēs and Palula dẽs day. The survey des responses fit that day family.',
'9980':'CDIAL 9980 *mahyā explicitly gives L. mañjh/majjh buffalo and a competing derivation through mahiṃśī. The survey many/mãny reflects the local palatal treatment, preserving uncertainty in the deeper reconstruction.',
'3906':'CDIAL 3906 khura explicitly gives Bshk khur and Torwali khū foot, while section 2 *khuḍa contains retroflex khuṛ. Exact parent selection respects the survey rhotic distinction.',
'2333':'CDIAL 2333 *uppari explicitly gives upar/uppar/opare above. The survey uppaṛ/upor forms fit this adverb; apar needs distinguishing apara.',
'8376':'CDIAL 8376 *peṭṭa gives peṭ/pẽṭ and explicitly aspirated peṭh belly. These e-vowel belly forms select the first branch.',
'3120':'CDIAL 3120 kāṣṭha gives kāṭh/kāṭhī wood and fuel. The simple kāth/kathi firewood forms fit; kaṭhua needs a suffix or adjective analysis.',
'10203':'CDIAL 10203 mudrā gives L. mundrī, Nepali mun(d)ro and regional mũdrī ring. These forms fit the ring family, whose remote Iranian origin Turner marks probable.',
'9369':'CDIAL 9369 bhaṇṭākī gives Bi. bhaṇṭā, B. bhā̃ṭā and Sinhala baṭu eggplant. The survey baṭa/bhaṭaa forms fit that first branch, retaining source nasalization or its absence.',
'6065':'CDIAL 6065 truṭyati explicitly includes Bhojpuri ṭūṭal and MIA tuṭṭa broken. The t-initial broken forms fit; initial d in duṭi requires checking the local sound or reading.',
'5994':'CDIAL 5994.2 *trāyaḥ explicitly contains Palula trō three, so these survey tro responses are assigned to that branch rather than the base trayaḥ.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for key in ('accepted','held') for x in d[key])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=r['Form'];target=k;reason=None
 if r['ID'] in done:continue
 if k=='10539' and w.startswith('a'):reason='Prefixed red form needs ārakta comparison.'
 elif k=='7150' and w=='nek':reason='Bare nek good needs immediate Persian donor versus nikta-family comparison.'
 elif k=='11745' and 'ŋ' in w:reason='Velar nasal in lightning requires local sound or transcription evidence.'
 elif k=='8249' and ('̃' in w or 'ũ' in w or 'ĩ' in w or w=='pāc'):reason='Nasal tail form needs the *puñcha subsection; a-vowel pāc additionally needs local phonology.'
 elif k=='11348' and w.startswith('g'):reason='Initial g in stone needs a distinct root comparison.'
 elif k=='6658-2' and r['Language_ID']=='Kal':reason='Short twelve response needs a compound or paradigm analysis.'
 elif k=='4368' and w.endswith('p'):reason='Village final p needs source or local phonetic verification.'
 elif k=='4225':target='4225-2'
 elif k=='6943':reason='River form needs nad/nadī or independent initial/final-consonant analysis.'
 elif k=='10990' and r['Language_ID']=='Dang':reason='Extra r in larasun garlic needs morphology or transcription analysis.'
 elif k=='13415' and r['Language_ID']=='Bshk':reason='Nīn river needs comparison with nadī, not an assumed loss of initial s.'
 elif k=='3906' and 'ṛ' in w:target='3906-2'
 elif k=='2333' and w=='apar':reason='Apar above needs distinction from apara.'
 elif k=='3120' and w=='kʌʈʰua':reason='Extended firewood form needs the -ua morphology checked.'
 elif k=='6065' and w.startswith('d'):reason='Initial d in broken needs local correspondence or source review.'
 elif k=='5994':target='5994-2'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires checking the lexical reading.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=36));continue
 c=x['comparanda'][k][0]['record'];ev=E[k]+' Reviewed comparison: '+c['Language_ID']+' '+c['Form']+' “'+c['Gloss']+'” ('+c['ID']+'). Possible intra-IA transfer remains open; source forms and provenance are preserved.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'near-sixth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-sixth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_sixth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','near-sixth'))
print('accepted',len(acc),'held',len(held))
