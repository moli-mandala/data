import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'3167':'CDIAL 3167 *kiyatta gives Gawri kata, Palula katī, Hindi kittā, and explicit -n and -r extensions kitnā/kitrā. These simple quantity-question forms fit those attested extensions; unexplained loss of t or added aspiration is left for local review.',
'3197':'CDIAL 3197 kīdṛśa gives kiso/kis and the analogically remodelled kaïsa/kaisā series. These simple s-bearing what-kind forms fit that interrogative family. Forms without s need separate comparison with other interrogative bases.',
'3832':'CDIAL 3832 kharva gives L./P. khabbā left, and explicit -ṭ extensions with xōṛi/khauṛi. The aspirated kh-bearing left forms fit those regional variants; unaspirated kab- is kept separate from the competing ḍābha family.',
'5679':'CDIAL 5679 tapta gives northern tat/tatt/tāto hot and an explicit -ll extension tātal/tātalā. The t-final hot forms fit these documented continuations; tapo and talo need distinguishing verbal or lateral developments.',
'6849':'CDIAL 6849 distinguishes dhūma smoke, with dhuā̃/dhuwā̃, from section 2 dhūmikā, expressly containing Palula dhūmī. The exact parent is chosen accordingly. Forms with unexplained k or intervocalic h need separate comparison.',
'3865':'CDIAL 3865 khādati gives khā-/khāv-/khāvan eat, including Palula khūm. Section 2 khādita expressly includes Nepali khāyo ate; the Dotyali past is assigned to that participle. Unclear p-bearing responses are held.',
'10648':'CDIAL 10648 raśmi gives rassī/rasrī/rasarī rope, notes initial l forms and possible rajju influence, and the addendum gives rɔśṭɔ. Northern rāz/rāy needs distinguishing rajju rather than a one-edit match.',
'9502':'CDIAL 9502 *bhiyajyate explicitly gives bhij-/bhīj- get wet and Mth. bhijlāh wet. The simple survey stem, infinitive and participial variants fit this verbal family; Dotyali -eko is an ordinary local participle on this identified wet verb.',
'12772':'CDIAL 12772 śvitra explicitly groups L./P. ciṭṭā white and J. ciṭā, while recording a citra alternative. The survey cit- white forms are provisionally grouped here with that deeper etymological uncertainty retained.',
'9209':'CDIAL 9209 *bāppa father lists bāp/bappā; section 2 *bābba has the voiced bab/bāb series. The unambiguous bāp forms fit the first branch; bop and pappā require nursery-family and voicing comparison.',
'14028':'CDIAL 14028 *hastakūṭa gives hathauṛā/hathoṛā and feminine hathoṛī hammer, with Gujarati athoṛī and Marathi hatoḍā. These full responses fit that compound family; isolated toḍī requires an independent analysis.',
'13291':'CDIAL 13291 *savēla gives Hindi sawerā morning and Ku./Nepali saber, explicitly explaining b through emphasis or ber. The suber-/saver- morning variants fit; nasal savā̃ra needs distinguishing *savāra.',
'6261':'CDIAL 6261 *dādda is a nursery kin term expressly including elder brother in several languages. Simple dad-/dod- elder-brother forms fit; feminine dādī in this gloss, initial-vowel aḍḍa and dat- need local review.',
'3244':'CDIAL 3244 kuṭhāra gives kuhāṛī/kuvāṛī axe, with metathesis explicitly discussed and kurhāṛi attested. These survey k/q forms fit those documented variants; initial g needs separate support.',
'6835':'CDIAL 6835 *dhūḍi/dhūli gives dhūr/dhūl/dhulo dust, northern duṛi and discusses variation dh/t. These simple dust responses preserve that family with the source aspiration and retroflex details retained.',
'1111':'CDIAL 1111 āṇḍa includes aṇḍā/āṇā egg and deformations with ṭ or lateral articulation. The uncomplicated nasal-stop forms fit; rhotic, initial h or additional -ī patterns are kept for their own branch/morphology review.',
'9964':'CDIAL 9964 mahiṣa lists bhaĩs/bhaĩso and mhaĩs buffalo, explicitly acknowledging some regional Hindi transfers. Survey bhæs/bhāiso/mhes variants fit that family; uncertain intra-IA transfer remains open.',
'9757':'CDIAL 9757 matsara mosquito lists macchar/macchur/macharu. These affricate and final-r variants fit that mosquito family; Turner’s uncertainty about its deeper Sanskrit history is retained.',
'142':'CDIAL 142 accha clear/good lists acchā/acho and explicitly notes regional transmission. These simple good responses fit that family, with variable aspiration and gemination preserved.',
'11165':'CDIAL 11165 lohita blood lists L. lahū/awāṇ lāū, regional lūī/loī, and expressly Kho. lei in its addendum. The survey lateral/rounded-vowel forms match these blood comparanda.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for key in ('accepted','held') for x in d[key])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=r['Form'];target=k;reason=None
 if r['ID'] in done:continue
 if k=='3167' and (w in {'kānɔ','kotka'} or 'ʰ' in w or w=='kitʰa'):reason='Quantity interrogative requires the aspiration, missing t, or extra k to be accounted for locally.'
 elif k=='3197' and w in {'keḍɔ','kayā','kaśyā','kāsam'}:reason='The altered interrogative stem needs distinguishing kīdṛśa from competing bases or formations.'
 elif k=='3832' and not ('ʰ' in w or w.startswith('kh')):reason='Unaspirated left stem needs resolving kharva versus ḍābha or another local formation.'
 elif k=='5679' and w in {'tapo','tʌlo'}:reason='Hot form needs distinguishing the tap verbal base or a lateral development from tapta.'
 elif k=='6849':
  if r['Language_ID']=='Phal':target='6849-2'
  elif r['Language_ID'] in {'Mai','Dotyali'}:reason='Smoke form has additional k or h requiring distinction from other smoke formations.'
 elif k=='3865':
  if r['Language_ID']=='Dotyali':target='3865-2'
  elif w=='kʰāpɔ':reason='The p-bearing eat response needs morphological identification.'
 elif k=='10648' and r['Language_ID'] in {'Mai','Phal'}:reason='Northern rāz/rāy rope needs comparison with rajju, not just raśmi.'
 elif k=='9209' and w in {'bop','pappā'}:reason='Father nursery form requires bāppa/bābba or independent papa comparison.'
 elif k=='14028' and w=='tɔḍī':reason='Hammer form lacks the first compound member; an independent tool-word derivation is possible.'
 elif k=='13291' and '̃' in w:reason='Nasalized savā̃ra morning needs comparison with *savāra.'
 elif k=='6261' and w in {'ʌdːa','dʌta','dādī'}:reason='Elder-brother form requires initial-consonant, kin-gender or t/d nursery-family analysis.'
 elif k=='3244' and w.startswith('g'):reason='Initial g in axe needs a local correspondence before choosing kuṭhāra.'
 elif k=='1111' and w in {'ãɳɽa','hānḍa','aṇḍāī'}:reason='Egg form requires rhotic-branch, initial h, or suffix analysis independently of the simple āṇḍa family.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires checking the lexical reading.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=34));continue
 c=x['comparanda'][k][0]['record'];ev=E[k]+' Reviewed comparison: '+c['Language_ID']+' '+c['Form']+' “'+c['Gloss']+'” ('+c['ID']+'). Possible intra-IA borrowing remains open under the user’s policy; exact source forms are retained.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'near-fourth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-fourth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_fourth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','near-fourth'))
from collections import Counter
print('accepted',len(acc),'held',len(held));print(Counter(x['parent'] for x in acc))
