import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'4971':('4971','CDIAL 4971 *chatti explicitly gives awāṇ chat and Lahnda/Punjabi chatt roof. These cat responses identify that roof noun; regional transmission remains open.'),
'2462':('2462-2','CDIAL 2462.2 *ēkka explicitly includes Bashkarik akh, Shina ek(h) and Lahnda hekk/hikk one. These retained-k forms select section 2, including h-bearing variants; Turner leaves emphatic strengthening versus learned replacement open.'),
'9201':('9201','CDIAL 9201 *bājjara explicitly means millet, but these surveys label bājrā barley. The crop identification or source-column alignment needs checking before treating them as the same lexical sense.'),
'10431':('10431','CDIAL 10431 yava explicitly lists regional jau/ja/yō barley. The barley responses fit that crop word; records glossed millet need independent crop-identification review.'),
'6333':('6333','CDIAL 6333 divasa includes day and sun, dīs/dīh, and the dihāṛā series; it gives Old Marwari dhyāḍo and Gujarati dahāṛɔ. The h-bearing forms may reflect crossing with ahar and regional spread, not a regular local s-to-h change. These qualifications are retained for the day/sun responses.'),
'2854':('2854','CDIAL 2854 kartati establishes kāṭ- cut. The specific bite sense needs its own primary attestation or local semantic check; the old exact link does not by itself establish it.'),
'10452':('10452','CDIAL 10452 yāti explicitly gives Hindi jānā, Maithili jāeb and Old Marwari jāvaï go; ordinary imperative jā-/jo forms fit this verb. The article separately derives suppletive g- past forms from gata, which are not included here.'),
'1564':('1564','CDIAL 1564 ittham describes this/that deictic adverbs, not the interrogative k-initial where response. A katra/kutrá or related interrogative comparison is required; no ittham link is justified by this candidate.'),
'7766':('7766','CDIAL 7766 padga explicitly lists Gujarati pāg foot/leg and Hindi pag foot. These whole pag leg responses fit that family; Turner leaves single g and short a unexplained. A compound pagḍaṇḍi path needs separate components.'),
'4963':('4963','CDIAL 4963 chagala includes Bashkarik chēl, Bhojpuri chērī, Hindi cherī and Gujarati chāḷī goat in section 1. These contracted goat forms fit that section; retained-g Bengali chagol needs a learned-form check against the strengthened branch.'),
'10896/10896-5/f_tjekvd6whuwmy':('10896-5','CDIAL 10896 explicitly places Nepali haluko and Hindi halkā under the metathesized -kk extension, stored as 10896-5 *laghukk-. The survey halka forms fit this exact branch, with intra-IA transmission unresolved.'),
'7627':('7627','CDIAL 7627 pakṣa explicitly includes Ku. pākho roof and pā̃kh feather, plus Nepali/Bhojpuri nasal feather forms. Nasalization may reflect crossing with puṅkha. The whole pāk/pā̃kh responses fit the containing entry; n-extended pakhnā needs separate morphology.'),
'13982':('13982','CDIAL 13982 hariṇa, meaning section 2, expressly gives Bengali harin deer. The surveyed horin forms fit that animal word. No separate stored section-2 node exists, so the containing entry is cited precisely.'),
'8283':('8283','CDIAL 8283 purāṇa gives Bengali purāna, Assamese purāni and regional paraṇa old. These uncontracted old-adjective forms fit the family; shortened punna and purna need consonant-cluster/length review.'),
'3208':('3208','CDIAL 3208 kukkuṭa explicitly gives Oriya kukuṛā, Gujarati kukṛɔ/kukṛī and Hindi kukṛā/kukṛī cock/hen. The survey chicken forms match those masculine/feminine continuations of the expressive family.'),
'13066':('13066','CDIAL 13066 sakala gives sayala/saala whole. Retained-g sagḷā or initial-h forms need their strengthened branch and local changes checked; the old generic sakala edge is insufficient.'),
'8209':('8209','CDIAL 8209 pibati explicitly gives Hindi pīnā, Maithili piab and Gujarati pīvũ drink. Forms with -le may include a light verb rather than an inflection and are held for segmentation.'),
'9361-2':('9361-2','CDIAL 9361.2 bhagna explicitly gives Old Marwari bhāgaï runs and Hindi bhāgnā flee. Its prose discusses break/flee and a competing historical root which would have merged completely. These g-bearing run forms select section 2, unlike j-bearing bhāj- in section 1.'),
'12803/12803-2':('12803-2','CDIAL 12803.2 *ṣuvaṭ explicitly gives Bashkarik ṣɔ, Torwali ṣō and Kalasha ṣɔ six. Rounded ṣo forms select section 2 rather than the plain ṣaṣ or affricate *kṣaṭ branch.'),
'10991':('10991','CDIAL 10991 *laṣṭi gives Nepali lāṭho and Hindi/Bhojpuri lāṭhī stick, alongside MIA laṭṭhi. The survey laṭ(h)i/laṭṭhi forms fit the stick family; the remote lakuṭa/yaṣṭi crossing remains tentative.'),
'11175':('11175','CDIAL 11175 vaṃśa explicitly gives Bengali bā̃s and related bamboo forms. Nasal bãś responses fit section 1; plain bas still needs a local nasal-loss or transcription check.'),
'8393':('8393','CDIAL 8393 *pokka explicitly gives Bengali pokā and Oriya poka insect/worm. These whole insect responses fit that precise family.'),
'10250':('10250','CDIAL 10250 mūla explicitly includes Bengali mul and Gujarati mūḷ/mūḷī root. The whole l-bearing root forms select section 1, retaining the remote-origin qualifications; Khandesi mui needs a local l-loss check.'),
'6663':('6663','CDIAL 6663 dvāra addendum explicitly gives Kacchi bāyṇo and Gujarati bārṇũ door. These n-bearing western door forms belong to that stored family; rhotic weakening and regional transmission remain open.'),
'11742/11745':('11742','CDIAL 11742 vidyut explicitly gives Gujarati vīj and Punjabi bijj lightning. Bare bij/vij selects this noun; vidyullatā 11745 contains the distinct bijlī/bijurī extended series.'),
'3696':('3696','CDIAL 3696 kṣīra explicitly gives Oriya khira and northern chīr milk. These forms fit the milk family; survey ṣīr requires distinguishing Iranian šīr or a demonstrated local change.'),
'10049/9828':('10049','CDIAL 10049 mānuṣa explicitly gives Gawri mānuṣ, while 9828 manuṣya lists short-vowel manuṣ forms and collisions. Gawri mānūṣ selects 10049; other insufficiently diagnostic forms remain under comparative review.'),
'3697':('3697','CDIAL 3697 kṣīraka expressly gives Nepali khiro and Hindi khīrā cucumber. These whole khira cucumber responses fit the precise cucumber entry, distinct from kṣīra milk.'),
'8141':('8141','CDIAL 8141 *pāhāḍa explicitly gives Bengali pāhāṛ hill/mountain. These eastern pahar responses fit that family; the deeper pāṣāṇa connection remains Turner’s qualified comparison.'),
'12159/12159-2':('12159-2','CDIAL 12159.2 *viyaṅga explicitly lists Assamese beṅ, Bengali beṅ/byāṅ and regional beṅ frog. These frog responses select section 2, not base vyaṅga speckled.'),
'134/135':('135','CDIAL 135 aṅguli explicitly gives Bengali āṅul/āṅguli finger. The anatomical finger gloss selects this stem rather than 134 aṅgula, a finger-breadth measure.'),
'9240':('9240','CDIAL 9240 bindu describes drops/spots and bindī dot, not needle. The Hajong bindi needle record requires a distinct etymology or sense/source check; it is not supported by the exact old link.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']+['loan-fifth-decisions.json']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ('accepted','held') for x in d[k])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 key='/'.join(x['parents']);r=x['record'];w=r['Form']
 if key not in E or r['ID'] in done:continue
 target,ev=E[key];reason=None
 if key in {'9201','2854','1564','13066','9240'}:reason=ev
 if key=='10431' and 'millet' in r['Gloss']:reason='Yava barley match has millet gloss; crop/source identification needs review.'
 if key=='7766' and 'path' in r['Gloss']:reason='pagḍaṇḍi path is a compound requiring separate ordered components.'
 if key=='4963' and ('g' in w):reason='Retained-g chagol goat needs learned chagala versus strengthened *chaggala review.'
 if key=='7627' and ('n' in w):reason='Feather pakhnā/pakna needs its nasal suffix distinguished from simple pakṣa/pā̃kh.'
 if key=='13982' and 'ŋ' in w:reason='Hajong horiŋ deer needs local final n/ŋ evidence.'
 if key=='8283' and w in {'punna','purna'}:reason='Contracted punna/purna old needs local cluster/length evidence before purāṇa assignment.'
 if key=='8209' and ('l' in w):reason='pile/pilija drink may contain a light verb or causative morphology; segment before linking.'
 if key=='9361-2' and w=='bāgɔ':reason='Bagri unaspirated bāgɔ run needs local deaspiration evidence.'
 if key=='11175' and w=='bas':reason='Hajong bas bamboo needs a local nasal-loss or transcription check.'
 if key=='10250' and w=='mui':reason='Khandesi mui root requires a local l-loss check.'
 if key=='3696' and (w.startswith('ṣ') or w=='kir'):reason='Milk ṣīr/kir needs local aspiration/sibilant evidence and Iranian šīr comparison where applicable.'
 if key=='10049/9828' and r['Language_ID']!='Gaw':reason=ev
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading verification.'
 if (key,target) not in ix:ix[key,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+(', meaning 2' if key=='13982' else '')+']',evidence=ev))
 i=ix[key,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=45));continue
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev+' Exact survey form '+w+' is preserved; uncertain intra-Indo-Aryan transmission remains open.'))
(P/'global-ninth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-ninth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_ninth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','global-ninth'))
print('accepted',len(acc),'held',len(held))
