import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'5428/5428-2':'CDIAL 5428 distinguishes ṭaṅka with L./P. ṭaṅg and Nepali ṭāṅ from section 2 ṭaṅga with Bengali, Mth., Bhojpuri, Hindi, Gujarati and Marathi ṭāṅg. The branch is selected by the explicit regional comparisons, with intra-IA transmission unresolved.',
'5536/5536-3':'CDIAL 5536.3 *ḍilla explicitly gives Gujarati ḍīl body, Sindhi ḍīlu body/belly and related bulk words. Retroflex ḍil body selects this branch, not the a-vowel *ḍala or dental *dilla.',
'4971/4981':'CDIAL 4971 *chatti expressly lists chatt/chat/chāt roof across the region. CDIAL 4981 chadman instead gives Bengali chād and nasal/voiced alternatives. These t-final roof forms select *chatti.',
'10158/10174':'CDIAL 10174 explicitly says the muk/mukh face forms more probably reflect mukha with expressive doubling than mukhya. Following that primary preference, these simple face/mouth forms are linked to 10158, leaving learned and intra-IA transmission open.',
'10896/10896-5':'CDIAL 10896 explicitly lists the -kk extension with metathesis halkā/haluko/haluk light. These k-bearing forms select the stored *laghukk- extension, not unextended laghu.',
'2668/2668-2':'CDIAL 2668.2 kaṇṭaka expressly gives Or. kaṇṭā, L./P. kaṇḍā, jaun. kā̃ḍā and kṭg. kaṇḍɔ thorn. These voiced or final-vowel thorn forms select that branch.',
'941':'CDIAL 941 aṣṭā/aṣṭau explicitly gives B. āṭ, Kalasha aṣṭ and Khowar oṣṭ eight. These dental/retroflex, rounded and nasalized variants identify that numeral family.',
'3245':'CDIAL 3245 *kuḍa explicitly includes feminine kuṛī girl/daughter/woman across L./P. and Palula. The survey girl/daughter responses fit that feminine series; the remote substrate origin remains qualified.',
'1634':'CDIAL 1634 ucca expressly includes above/high and ucā/uñcā forms. The c-bearing above responses fit this family; s-bearing forms require local deaffrication evidence.',
'10805':'CDIAL 10805 rūpya explicitly gives Bengali/Or. rupā and widespread rūp/rūpo silver. These simple rup- silver forms select the first branch, not the contracted *rūpiya series.',
'12193':'CDIAL 12193 vyāghra explicitly gives B./Assamese bāgh/bāg tiger. The survey bagh forms fit that family; Hajong bak needs local final-devoicing evidence.',
'2771':'CDIAL 2771 kambala expressly gives B. kambal blanket and records regional borrowing from the northwest. These kombol/kɔmbol forms fit that blanket family with the transmission route left open.',
'7621':'CDIAL 7621 pakva explicitly gives Bengali pākā ripe and corresponding regional forms. These a-vowel ripe responses select the first branch, distinct from *pikva.',
'10738':'CDIAL 10738 riṅgaṇī explicitly gives Gujarati/Marathi riṅgṇī eggplant and the fruit nouns rĩgṇũ/rĩgṇẽ. These western eggplant forms preserve that diagnostic nasal-velar sequence; possible Gujarati/Marathi mediation remains open.',
'13095':'CDIAL 13095 sajya expressly gives L./P. sajjā righthand, including awāṇ. sajjā. These saj- right forms fit that family, not dākṣiṇa.',
'10302':'CDIAL 10302 megha expressly gives awāṇ. mī̃, L. mī̃h and Hindi mẽh rain. These simple mi/mī̃/meh responses preserve that rain family.',
'9297-2':'CDIAL 9297.2 *būṭṭa explicitly gives L./P. būṭā plant and WPah. buṭṭ tree. These būṭā/buṭa tree responses select section 2; the ultimate relation to Persian bōta remains open.',
'2998':'CDIAL 2998 *kākka expressly gives Nepali kāko and B./H. kākā father’s younger brother. The survey kin term matches this exact relationship without collapsing other uncle terms.',
'3317':'CDIAL 3317 kumbhīra explicitly gives B. kumīr and Assamese kumbhīr crocodile. These kumir responses fit that animal family.',
'9397':'CDIAL 9397 bharati includes bharita full and Bhojpuri bharal fill. The survey bharal/bhora full forms fit ordinary participial continuations of that fill verb.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']+['near-sixth-decisions.json']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for key in ('accepted','held') for x in d[key])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 key='/'.join(x['parents'])
 if key not in E:continue
 r=x['record'];w=r['Form'];target=key.split('/')[0];reason=None
 if r['ID'] in done:continue
 if key=='5428/5428-2':target='5428' if r['Language_ID'] in {'Goj','awan','jaun'} else '5428-2'
 elif key=='5536/5536-3':target='5536-3'
 elif key=='4971/4981':target='4971'
 elif key=='10158/10174':target='10158'
 elif key=='10896/10896-5':target='10896-5'
 elif key=='2668/2668-2':target='2668-2'
 elif key=='1634' and ('s' in w or 'ś' in w):reason='S-bearing above form needs local c/s correspondence.'
 elif key=='12193' and w=='bak':reason='Final k in tiger requires a local devoicing check.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires checking the lexical reading.'
 if (key,target) not in ix:ix[key,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[key]))
 i=ix[key,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=37));continue
 c=x['comparanda'][0]['record'];ev=E[key]+' Discovery comparator: '+c['Language_ID']+' '+c['Form']+' “'+c['Gloss']+'” ('+c['ID']+'). The full primary entry was reviewed independently of the existing match. Exact survey forms are preserved.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'global-sixth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-sixth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_sixth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','global-sixth'))
print('accepted',len(acc),'held',len(held))
