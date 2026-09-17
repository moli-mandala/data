import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'12803/12803-3':('12803-3','CDIAL 12803.3 *kṣaṭ explicitly contains Punjabi che, Nepali cha, Oriya cha and Hindi/Gujarati cha; the addendum gives Jaunsari chau. These affricate six forms select section 3, not the ṣaṣ or *ṣuvaṭ branches.'),
'6955b':('6955b','CDIAL 6955b náptṛ explicitly gives Nepali nāti grandson and nātini granddaughter, with Bhojpuri nātin and related feminine derivatives in the same article. The surveyed kin terms preserve those masculine/feminine forms.'),
'13952':('13952','CDIAL 13952 haḍḍa lists Oriya hāṛa, Hindi haḍḍī, Gujarati hāḍ and Marathi hāḍ bone; the addendum explicitly includes hāṛkɔ and hāḍkī. These bone forms fit that family, including k-extended forms. The remote connection to asthi is explicitly very doubtful.'),
'11363/11366':('11366','CDIAL 11366 vartman explicitly lists Gujarati, Marathi and Konkani vāṭ path. It also acknowledges that some feminine forms could equally derive from vartis (11363). Following the primary regional placement, vartman is provisionally selected; the vartis alternative remains unresolved, not disproved.'),
'3471':('3471','CDIAL 3471 keśa explicitly gives Nepali/Hindi kes and Gawri khẽs hair. The exact kes/keś family fits, with contact transmission unresolved.'),
'4749':('4749','CDIAL 4749 *cāmala or *cāvala explicitly gives Lahnda/Punjabi cāval, awāṇ cāvul and Bhojpuri/Maithili cāur husked rice. These survey responses match that family; the ultimate non-Aryan source remains unsettled.'),
'10896-5':('10896-5','CDIAL 10896 gives awāṇ laùkā and regional lōka forms under the -kk extension, and Nepali haluko under its metathesized series. Both belong to the stored *laghukk- section 10896-5.'),
'5481':('5481','CDIAL 5481 *ṭoppa explicitly gives Nepali ṭopi hat/cap and Hindi/Bhojpuri ṭopī. These whole hat responses match that feminine noun, with regional transmission unresolved and remote connections left as Turner’s doubtful alternatives.'),
'6065':('6065','CDIAL 6065 truṭyati explicitly includes Prakrit tuṭṭa broken, Hindi ṭūṭā and Bhojpuri ṭūṭal. The survey ṭuṭā/ṭuṭo/ṭuṭel forms are participial continuations of that verb.'),
'4445':('4445','CDIAL 4445 gharma explicitly gives Assamese/Bengali ghām heat/sweat and Hindi ghām heat/sunshine/sweat. The survey ghām sweat forms and Dangaura hot sense fit the same heat family.'),
'700':('700','CDIAL 700 alagna explicitly lists Hindi/Nepali alag and regional alag/ālag separate. The different responses include transparent distributive repetition alag-alag; both repeated parts continue this same lexeme, not an unrelated second component.'),
'13574/13574-3':('13574-3','CDIAL 13574.3 sūriya explicitly lists Gawri sūrī, Kalasha sūri, Palula sūri and Shina sūri sun. These retained-r sūri forms select section 3; the different Maiya swīr form is placed under section 4 in Turner, so transmission into surveyed Maiya sūri remains open.'),
'12548':('12548','CDIAL 12548 śuṣka lists Hindi sūkhā and Gujarati sūku dry, and its -ll extension lists Oriya sukhilā. The l-bearing dry forms are assigned to the specific stored extension 12548-2; other phonological or verbal extensions require separate review.'),
'10191/10247':('10247','CDIAL 10247 mūrdhan discusses the head words explicitly but says unaspirated forms may be from or crossed with muṇḍa shaven (10191). The candidate match alone cannot decide these competing sources.'),
'5994-3':('5994-3','CDIAL 5994.3 trīṇi contains dental tin/tīn and Gujarati traṇ. The survey initial retroflex ṭ needs a demonstrated local correspondence before accepting that precise analysis.'),
'5589':('5589','CDIAL 5589.1 *ḍhiḍḍha explicitly gives Lahnda/Punjabi ḍhiḍ(ḍh) belly. Initial ṭ and e-vowel responses differ from that exact branch and need local evidence or another subsection.'),
'6726':('6726','CDIAL 6726 dhanus means bow, but the survey sense is rainbow and its retained final ś/s suggests learned or regional transmission. A whole-word donor or independent lexical evidence for that sense is needed before copying the old Sanskrit link.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ('accepted','held') for x in d[k])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 key='/'.join(x['parents']);r=x['record'];w=r['Form']
 if key not in E or r['ID'] in done:continue
 target,ev=E[key];reason=None
 if key=='3471' and r['Language_ID']=='Bhilali' and 'kh' in w:reason='Bhilali aspirated khes needs local aspiration evidence; Gawri khẽs does not establish the Bhil correspondence.'
 if key=='12548':
  if 'l' in w:target='12548-2'
  else:reason='Dry form needs local h/s correspondence or verbal -y- analysis; not resolved by the exact match.'
 if key=='10191/10247':reason=ev
 if key=='5994-3':reason=ev
 if key=='5589' and not w.startswith('ḍʰi'):reason=ev
 if key=='6726':reason=ev
 if key=='700' and ('nar' in w or 'ñar' in w):reason='narā/ñara different requires a separate nyāra analysis, not alagna.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading verification.'
 if (key,target) not in ix:ix[key,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=ev))
 i=ix[key,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=41));continue
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev+' Exact survey form '+w+' is retained; possible cross-Indo-Aryan transmission does not exclude this lexical-family link.'))
(P/'global-seventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-seventh-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_seventh_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','global-seventh'))
print('accepted',len(acc),'held',len(held))
