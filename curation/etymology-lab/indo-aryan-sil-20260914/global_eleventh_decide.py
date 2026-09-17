import json,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'3153':('3153','*kicca explicitly includes Palula kičal/číčal and Gujarati kīcaṛ mud; these cīcal/kicaḍ forms match the documented extensions.'),
'1111':('1111','āṇḍa explicitly includes Palula haṇo, Torwali āṇ and Gujarati ĩḍũ egg. The initial-h and raised/nasalized Bhil forms therefore have direct comparative support.'),
'6648':('6648','dva includes dvi in Niya and retroflex ḍu forms in Sindhi/Lahnda. Western ḍo/ḍū and Bareli dvi fit the family; Dangaura retroflexion needs a local check.'),
'8249-2':('8249-2','Section 2 *pucchaḍa explicitly includes Nepali puchar, Gujarati puchṛũ and northern puchaṛ tail; the r-bearing survey forms select this extended branch.'),
'9226':('9226','*bāhira explicitly includes bāhar/bahir/bahara outside. These survey adverbs fit the documented regional variants.'),
'2575':('2575','kaḥ punar explicitly includes Nepali kun and Oriya kun which. Survey kun/kon interrogatives fit this family, with the source who/which distinction retained.'),
'12115':('12115','velā supplies time/daytime, but the article does not directly establish the Indo-Aryan bela sun sense; the Dravidian sun comparison alone cannot establish this survey link.'),
'3952':('3952','gaṅgā explicitly includes Assamese gāṅg/gāṅ and Bengali gāṅ river. Hajong gaŋ river matches this generalized river name; regional transmission remains possible.'),
'13544':('13544','sūkara includes suvar/suor/sūr pig in its first branch. Survey suvar/sur/suor/sõra fit that branch; initial-h variants need a local correspondence check.'),
'3770':('3770','khaṭakkikā explicitly includes Nepali khirki and Bengali/Assamese window forms. These khirki responses match this window family.'),
'9377/9408':('9408','bhalla explicitly gives Bengali bhāla and Oriya bhala good. The l-bearing Hajong forms select 9408, not the competing bhadra entry 9377.'),
'1268':('1268','āmra explicitly gives regional ām/amb mango. Bhatri and Halbi ama mango fit the documented mango family.'),
'4822':('4822','*cimb concerns pinching and tongs and does not establish these caṭi/čẽṭi ant forms. The old comparative assignment is insufficient; the insect family must be identified independently.'),
'9250/9261':('9250','Retained-j bij seed can reflect learned bīja or the bījya branch. The eastern survey forms require this distinction to be resolved rather than automatically following an old comparative link.'),
'4089':('4089','galla explicitly gives Bengali gal cheek. These Bengali/Bishnupriya responses belong to the cheek family, distinct from gala throat/neck at 4070.'),
'9822':('9822','manas mind is a plausible family, but the heart sense needs explicit lexical evidence and its organ versus emotional sense checked before accepting these responses.'),
'10437':('10437','yavākāra explicitly includes juvar/juar millet. These survey forms identify that cereal family, preserving the source generic millet label.'),
'6983':('6983','nava explicitly includes Gujarati navũ and WPah. nov new. The y-bearing forms still compete with navya, and Dogri nama needs a local nasal-development check.'),
'10984':('10984','*lavaṇḍa explicitly gives Hindi lauṇḍā boy and Kumaoni lauṛo without a nasal. Survey lauḍa/lauḍiya/loɳɖa/laura fit this family, retaining ordinary feminine and son/boy distinctions.'),
'9465':('9465','bhārika explicitly includes weighty, heavy, big and much in the addendum. Tharu bhari big is therefore semantically documented, not inferred solely from weight.'),
'4345':('4345','gaura explicitly includes Kalasha gɔ̈ra, Nepali goro and Gujarati gorũ white. Simple gora/goro match this colour family; gorahar needs separate extension analysis.'),
'10323':('10323','medas explicitly includes Kalasha mẽ, Torwali mih and related Dardic fat nouns. Survey fat needs its noun/adjective elicitation sense checked before accepting these very short forms.'),
'7467':('7467','The full niṣīdati article explicitly places Khowar and Gawri niś forms as loans from Nuristani. A direct inherited link would erase that route; identify the supported immediate donor first.'),
'4701':('4701','carman explicitly includes Hindi camṛā, Gujarati cāmḍī and Lahnda/Punjabi camṛā skin/leather. The c/ts forms fit those extensions; regional s forms need their own affricate/fricative correspondence evidence.'),
'13992':('13992','haridrā explicitly includes Lahnda hardal, Punjabi haldī and Shina hălĭẓi turmeric. These variants fit branch 1, with regional contact possible; contracted hād needs a local liquid-loss check.'),
'13845':('13845','*sphuṭyati is a finite burst/break verb family. These broken responses include participles, aspiration shifts and phuṭgaya with a second verb; their exact morphology and parent need review.'),
'14028':('14028','*hastakūṭa explicitly includes Hindi hathauṛī and regional hathaurā/hātuṛi hammer. These hathauri responses match the whole hand-hammer compound.'),
'13366-2':('13366-2','Section 2 sārthika explicitly includes Nepali/Bengali sāthi and regional sāthī companion/friend. These forms select this branch rather than unextended *sārthin.'),
'10299':('10299','mṛṣṭa/miṣṭa explicitly includes eastern miṭhā sweet and Marathi mīṭh, Konkani mīṭa salt. Hajong sweet and Khandesi salt therefore fit the same documented semantic family.'),
'14615/6495':('6495','dūra explicitly includes Bengali dur, Oriya dura and regional dūri adverbial forms. Dure far fits this family; 14615 is an addendum to the same head, not a competing root.'),
'10552/10555':('10555','*rakṣāpuṭaka explicitly includes Gujarati rākhɔṛɔ/rākhɔṛī ashes. The survey rakhoḍo/rakhoḍi forms retain the extended compound ending and select 10555 rather than simple rakṣā 10552.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
acc=[];held=[];rules=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 k='/'.join(x['parents']);r=x['record'];w=r['Form']
 if k not in E or r['ID'] in done:continue
 parent,ev=E[k];ev='CDIAL '+parent+' '+ev;reason=None
 if k in {'12115','4822','9250/9261','9822','10323','7467','13845'}:reason=ev
 if k=='6983' and w not in {'navũ','nov'}:reason=ev
 if k=='4345' and r['Language_ID']=='Chitwan':reason='Gorahar white requires identifying its extra suffix or expressive morphology.'
 if k=='6648' and r['Language_ID']=='Dang':reason='Dangaura ḍu/ḍui requires local retroflexion evidence; the western ḍu comparison alone is insufficient.'
 if k=='8249-2' and w=='pusəḍu':reason='Bhilali pusəḍu tail requires local cʰ/s and suffix evidence before choosing the extended branch.'
 if k=='13544' and w.startswith('h'):reason='Hajong huvar pig requires confirming the local s/h correspondence.'
 if k=='4701' and w.startswith('s'):reason='Skin form with s requires a regional c/s correspondence check.'
 if k=='13992' and w=='hād':reason='Gujari hād turmeric requires confirming liquid loss or its source transcription.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires checking the exact lexical reading.'
 if k not in ix:ix[k]=len(rules);rules.append(dict(parent=parent,citation='CDIAL['+parent.replace('-','.',1)+']',evidence=ev))
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=50))
 else:acc.append(dict(record=r,family=i,parent=parent,citation=rules[i]['citation'],evidence=ev+' Exact survey form '+w+' is preserved; intra-Indo-Aryan transmission remains open.'))
(P/'global-eleventh-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1))
(P/'global-eleventh-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
(P/'global_eleventh_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-eleventh'))
print(len(acc),len(held))
