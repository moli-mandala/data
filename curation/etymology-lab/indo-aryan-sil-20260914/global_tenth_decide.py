import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'6835':('6835','CDIAL 6835 *dhūḍi includes Oriya dhuḷi and regional dhūl/dhūṛ dust. Simple dhuli/dhuḍ forms fit this family; reduplicated dhudhar needs separate morphology.'),
'11119':('11119','CDIAL 11119 loka gives people/world, not the surveyed singular man or husband. The retained-g log also needs the strengthened/learned form distinguished; an exact old link does not settle this use.'),
'9661':('9661','CDIAL 9661 bhrātṛ explicitly gives bhāi/bhāu brother, older/younger local distinctions, and Palula brhō. The survey bhāyo/bhaiyā/bro forms fit this kinship family, preserving the source age distinctions.'),
'10560':('10560','CDIAL 10560 raṅga explicitly gives Bengali rāṅā and Oriya rāṅgā red. The Hajong raŋa red responses fit this colour family.'),
'10990':('10990','CDIAL 10990 distinguishes l-initial laśuna from section 2 raśuna, explicitly including Bengali rasun. The survey initial consonant selects the branch, with regional vowel colouring retained.'),
'10250/10250-2':('10250','CDIAL 10250 mūla explicitly gives Gujarati mūḷ root, whereas *mūḍa section 2 concerns a distinct northern retroflex series. The Bhil muḷ responses follow the Gujarati root family, without treating retroflex l alone as proof of *mūḍa.'),
'5082':('5082','CDIAL 5082 jaṅghā explicitly gives Lahnda jaṅgh leg and Hindi jā̃g thigh/leg. These Pothwari jaŋ leg responses fit the shank/leg family, with final aspiration variation and local transmission left open.'),
'11250':('11250','CDIAL 11250 vadhū explicitly gives Hindi bahū bride/wife and Maithili bahu wife. The Khowar bok comparison has a specifically qualified *vadhukkā extension in the addendum and is held for exact-parent review.'),
'10931':('10931','CDIAL 10931.1 *lattā explicitly gives Lahnda/Punjabi latt leg and awāṇ lat. These leg responses select section 1 rather than *latthā kick.'),
'11572':('11572','CDIAL 11572 vāla explicitly gives Palula bōla/būla hair, Oriya bāḷa and WPah. bā. The survey bāl-/būl-/bā forms fit this hair family, with any regional transmission retained as uncertain.'),
'4780':('4780','CDIAL 4780 cikka explicitly gives Punjabi cikkaṛ, Hindi cīkaṛ and the kṭg. addendum cikṛɔ mud. These cikaṛ/cikaḍ forms fit the r/ḍ extension of this sticky-matter family; its remote origin is debated.'),
'10978':('10978','CDIAL 10978 lavaṇa explicitly gives Niṅg. lõ and regional lūṇ/lōn salt. Nasalized lõ/lū̃ fit contraction of this salt word; bare Gujari nu requires confirming nasal loss rather than relying on the old match.'),
'9964':('9964','CDIAL 9964 mahiṣa explicitly gives maīś/mẽṣ/maĩś and regional bhẽs buffalo. These sibilant-bearing variants fit the buffalo family; very reduced me and unexplained bhās remain separate review cases.'),
'9085':('9085','CDIAL 9085 *phutta explicitly gives Bashkarik phit mosquito and Palula phutti mosquito. Simple phīt responses fit that family; c-bearing forms or extra final aspiration require local sound evidence.'),
'10757':('10757','CDIAL 10757 *rukṣa gives Punjabi rukkh and eastern rūkh tree. Simple ruk is provisionally assigned to that family; extended/nasalized rũkəḍo forms need their own suffix analysis.'),
'11302':('11302','CDIAL 11302 vayam explicitly gives Maiya bē, Chiliss/Gowro be and Palula be/beh we. These survey be/beh forms select that pronoun family; the exact source number is retained.'),
'13544-2':('13544-2','CDIAL 13544.2 *sūṅkara explicitly gives Nepali sũgur domesticated pig. The survey suŋgur forms retain the diagnostic nasal-plus-velar sequence and select section 2.'),
'9504':('9504','CDIAL 9504 *bhiyantara explicitly gives Nepali bhitra, Maithili bhitrī and regional bhitar inside. These inside adverbs match that comparative family with its abhyantara-crossing qualification retained.'),
'2867/2869':('2869','CDIAL 2867 karda gives Bengali kādā/Oriya kāda, while 2869 kardama gives Bengali/Oriya kādo and regional kādau/kādā. Eastern o-final forms select kardama; the a-final and central forms remain insufficiently diagnostic.'),
'2871':('2871','CDIAL 2871 karpaṭa explicitly gives Assamese kāpar/kāpor garment and Kashmiri kapur clothes. These Hajong kapor/kapur clothing forms fit the regional cloth family, with the deeper analysis qualified in Turner’s addendum.'),
'7081':('7081','CDIAL 7081 nāvā explicitly gives Assamese nāu boat and related regional nāu forms. These Hajong nau boat responses fit that noun.'),
'10187/10187-11':('10187-11','CDIAL 10187.11 *moṭṭa explicitly gives Gujarati moṭũ and Old Marwari moṭaü big/fat, alongside Bengali moṭā fat. These o-vowel adjectives select section 11, not unextended *muṭṭa defective.'),
'10163':('10163','CDIAL 10163 mukhatuṇḍaka explicitly gives Gujarati mɔḍhũ/mɔ̃ḍũ mouth/face. The Bhil moḍu mouth/face forms fit that whole compound family with regional Gujarati transmission possible.'),
'2665':('2665','CDIAL 2665 kaṇika, replaced by *kaṇikka, explicitly gives Lahnda/Punjabi kaṇak and WPah. kaṇak wheat. These survey wheat forms fit that precise grain family.'),
'5267-2/5267-3':('5267-3','CDIAL 5267.3 *jimyati/*jimmati explicitly gives Hindi jīmnā and Old Marwari jīmaï eat. The survey jīmɔ forms select the strengthened third branch, not *jimati in section 2.'),
'3475':('3475','CDIAL 3475 kesarin refers chiefly to lion, with a Marathi adjective fibrous of mango. That does not establish keri mango as its descendant; a separate mango lexeme and source are required.'),
'13479':('13479','CDIAL 13479 supta explicitly gives Maithili sūtab sleep, Old Marwari sūto asleep and Palula suttu past slept. These t-bearing sleep forms select the participial family; initial-h forms require a local correspondence check.'),
'11567':('11567','CDIAL 11567 vārdala explicitly gives Hindi badlī and regional bādal cloud. Cloud responses fit; a whole sky gloss needs a local semantic or source-column check before the cloud word is assigned.'),
'9917':('9917','CDIAL 9917 maśaka explicitly gives Gawri masa, Bengali maśā and Hindi masā mosquito, with nasal variants elsewhere in the article. These mosquito responses fit that insect family.'),
'12778':('12778','CDIAL 12778 śvaitra explicitly gives Nepali seto and Old Awadhi seta white. These set/seto responses select this e-vowel family, distinct from śvitra alternatives.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ('accepted','held') for x in d[k])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 key='/'.join(x['parents']);r=x['record'];w=r['Form']
 if key not in E or r['ID'] in done:continue
 target,ev=E[key];reason=None
 if key in {'11119','3475'}:reason=ev
 if key=='6835' and w=='dʰudʰʌɾ':reason='Reduplicated/extended dhudhar dust needs separate morphology.'
 if key=='10990' and not w.startswith('l'):target='10990-2'
 if key=='11250' and r['Language_ID']=='Kho':reason='Khowar bok/bōk wife needs the *vadhukkā extension identified exactly, per Turner’s addendum.'
 if key=='10978' and w=='nu':reason='Gujari nu salt needs local final-nasal loss or transcription evidence.'
 if key=='9964' and w in {'me','bɦas'}:reason='Very reduced or unexpectedly vocalized buffalo form needs local phonological evidence.'
 if key=='9085' and w not in {'pʰīt'}:reason='Mosquito form has affrication or extra final aspiration not established by the phit/putti comparison.'
 if key=='10757' and w!='ruk':reason='Rũkəḍo/rukəḍo tree needs nasal and suffix analysis before assigning simple *rukṣa.'
 if key=='2867/2869' and not (w=='kado' and r['Language_ID'] in {'AdivasiOriya','Bhatri','Hajong'}):reason='Mud karda/kardama branches remain insufficiently distinguished for this form.'
 if key=='13479' and w.startswith('h'):reason='Marwari hūtɔ sleep needs the local s/h correspondence checked.'
 if key=='11567' and 'sky' in r['Gloss']:reason='Bādal cloud under a sky gloss needs local semantic/source alignment verification.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading verification.'
 if (key,target) not in ix:ix[key,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=ev))
 i=ix[key,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=46));continue
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev+' Exact survey form '+w+' is preserved; intra-Indo-Aryan transmission remains open.'))
SE={
'3167':'CDIAL 3167 explicitly lists kitnā, kittā, Marathi kitī and ketek how many. These spaced source spellings preserve the same complete lexical form.',
'10511':'CDIAL 10511 explicitly lists Hindi tum you, historically plural with t from tvam. The source spacing t ūm is internal to that single pronoun; its elicitation label is retained.',
'1577':'CDIAL 1577 indradhanuṣ is explicitly rainbow, composed of Indra and bow. These indra dhanuś responses match the complete learned compound despite the source word boundary.',
'13161':'CDIAL 13161 saptāha explicitly means a period of seven days. These spaced sāpt ā/sapt a responses retain the recognizable week word; learned or regional transmission remains open.',
'3197':'CDIAL 3197 explicitly gives Hindi kaisā and Apabhraṃśa kaïsa of what kind. Source kai so/kei so matches that single interrogative adjective.',
'13551':'Needle sv i needs a source reading check before assuming the v represents a glide or vowel in the sūcī family.',
'9661':'CDIAL 9661 includes bhāiya/bhāi brother and source-specific younger-brother uses. Spaced bhai yā is a single kin term rather than an added lexical component.'}
for x in json.loads((P/'spacing-candidates.json').read_text()):
 if len(x['parents'])!=1:continue
 k=x['parents'][0];r=x['record']
 if k not in SE or r['ID'] in done:continue
 i=len(qs);ev=SE[k];qs.append(dict(parent=k,citation='CDIAL['+k+']',evidence=ev))
 if k=='13551':held.append(dict(record=r,families=[i],reason=ev,passNumber=46));continue
 acc.append(dict(record=r,family=i,parent=k,kind='borrowed' if k=='1577' else 'reflex',citation=qs[i]['citation'],evidence=ev+' Whitespace was ignored for comparison only; the exact source form is unchanged.'))
(P/'global-tenth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-tenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_tenth_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-tenth'))
print('accepted',len(acc),'held',len(held))
