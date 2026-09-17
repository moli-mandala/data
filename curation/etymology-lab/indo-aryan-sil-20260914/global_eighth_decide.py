import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'9237':('9237','CDIAL 9237 biḍāla explicitly gives Nepali birālo, Assamese birali, Bengali bilāi and Maithili bilār cat. These expanded cat forms belong to section 1, distinct from the contracted *billa section.'),
'11242':('11242','CDIAL 11242 vatsara and its addendum explicitly give Assamese basar and Bengali bachar year. Turner questions whether these were borrowed from Sanskrit; that learned or regional transmission remains open.'),
'9415/9415-2':('9415-2','CDIAL 9415.2 *bhallukka explicitly contains Assamese/Bengali bhāluk bear. Retained final k selects this strengthened branch, not the bhālū forms under section 1.'),
'9459':('9459','CDIAL 9459 bhāra explicitly includes Punjabi bhārā heavy, alongside bhār load/weight. These bhār/bhāru/bhārā heavy forms fit the burden/weight family; no claim is made that every adjective preserves exactly the Sanskrit inflection.'),
'7918':('7918','CDIAL 7918 parṇa explicitly gives Hindi/Marwari pān and Gujarati pān/pā̃dṛũ leaf. These leaf responses, including the ṛ-extended series, select section 1; the competing pārṇa branch concerns betel leaves specifically.'),
'4855/4855-2':('4855-2','CDIAL 4855.2 *cucci explicitly lists Bashkarik čič and Hindi/Nepali cūcī/cuci breast or nipple. Their i-bearing series selects section 2, not *cuccu; the family is expressive.'),
'6481':('6481','CDIAL 6481 duhitṛ explicitly lists Maiya/Palula/Punjabi dhī and Oriya jhi daughter, with Assamese zī. The dh-/jh-/z- variants are discussed in the primary article; the abnormal developments and possible jātá influence remain qualified there.'),
'242':('242','CDIAL 242 adya explicitly gives Maiya āz, Torwali až-dī, Shina aš and Oriya āj/āji today. These āj/āz/āž/āś variants fit the same today family; any inter-language transmission remains unresolved.'),
'11493':('11493','CDIAL 11493 *vātatrāsa explicitly gives Nepali batās, Bengali bātās and Maithili both batās and basāt wind. The metathesized survey bəsat is thus supported within this exact wind family.'),
'2095':('2095','CDIAL 2095 distinguishes undura with u-vowel rat/mouse forms from section 2 indūra, explicitly represented by Bengali ĩdur and Assamese endur. The survey vowel identifies the selected branch; the ultimate Austroasiatic etymology remains Turner’s attribution.'),
'4070':('4070','CDIAL 4070 gala explicitly gives Bengali galā throat and Maithili/Bhojpuri gar neck/throat. The surveyed eastern gar/gala neck forms fit section 1, with local or contact-mediated l/r variation left open.'),
'8370':('8370','CDIAL 8370 pṛṣṭi explicitly lists Assamese/Oriya piṭhi back; the retained final i supports this feminine stem rather than the pṛṣṭha alternative Turner raises for forms without final i.'),
'5806':('5806','CDIAL 5806 tikta explicitly lists Assamese titā and Bengali tita/titā bitter. These whole bitter responses fit that family.'),
'7540':('7540','CDIAL 7540 nīca explicitly gives Gujarati nīcε and Old Marwari nīcaï below, with strengthening after opposite ucca. The c/ch-bearing survey forms fit that family; western s-bearing forms still need their local correspondence established.'),
'13386':('13386','CDIAL 13386 sikatā explicitly gives Gawri sīũ and an l-extension represented by Torwali sigəl, Chiliss/Gowro sigil and Shina sigal sand. The article explicitly leaves suffix variation and Iranian or intra-regional borrowing as alternatives. These forms are linked to the containing entry with that transmission uncertainty retained.'),
'12487':('12487','CDIAL 12487 śītala explicitly lists Gawri šalá cold, Hindi sīlā cool and Torwali/Palula šidul/šidālo with unexplained d. Its addendum qualifies l/lh forms as *śītalla or crossed with *śaitalya. The containing article is used with those phonological qualifications retained, not as a claim of regular unmodified inheritance.'),
'11497':('11497','CDIAL 11497.1 vātara explicitly lists Gujarati vāyrɔ and Marathi vārā wind. Section 2 vātala separately includes Bashkarik bālā and Hindi bāl; l-bearing survey forms are assigned to that specific section.'),
'9250':('9250','CDIAL 9250 bīja explicitly gives Nepali biu and plural biyā̃, Bengali biā and Bhojpuri bīyā seed. These seed responses fit that same family.'),
'6767':('6767','CDIAL 6767 dhavala explicitly gives Gujarati dhɔḷũ and Assamese/Bengali dhala white. Unaspirated d in the surveyed forms needs local deaspiration evidence, rather than being assumed from an exact old link.'),
'3103':('3103','CDIAL 3103 kāleyaka expressly discusses the heart/liver semantic overlap and gives Hindi karejā heart/liver, Assamese kɔlizā heart and Gujarati kāḷjũ heart/liver. The survey kalja/kaleja/kareja heart words fit this explicitly documented sense family.'),
'6582':('6582','CDIAL 6582 dola explicitly gives Prakrit ḍola eye, Oriya doḷā/ḍoḷā pupil and Marathi ḍoḷā eye. These western ḍoḷo/ḍoḷā eye forms match that eye branch; Marathi or other regional transmission remains open.'),
'28':('28','CDIAL 28 akṣata explicitly gives Gujarati ākhũ and Marathi ākhā whole, and Konkani ākho/āko complete. These western whole responses preserve that adjective family.'),
'7743/7785':('7743','CDIAL 7743 patha lists paha/pāhā forms, while 7785 panthā lists nasal panth/pand forms. Neither article establishes the survey eastern pot as a regular reflex; an immediate Assamese/Bengali learned path word needs checking.'),
'11392':('11392','CDIAL 11392 varṣa contains rain and year branches, but the survey barkhā forms need the separate feminine varṣā and its exact stored parent checked. Do not copy the generic old varṣa link.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ('accepted','held') for x in d[k])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 key='/'.join(x['parents']);r=x['record'];w=r['Form']
 if key not in E or r['ID'] in done:continue
 target,ev=E[key];reason=None
 if key=='9459' and w=='phəro':reason='Bhilali initial ph in heavy requires local bh/ph evidence; Romani pharo does not itself establish this Bhil development.'
 if key=='2095' and w.startswith('i'):target='2095-2'
 if key=='11497' and ('l' in w or 'ḷ' in w):target='11497-2'
 if key=='7540' and 's' in w:reason='Western nis- below needs the local c/s correspondence established before linking.'
 if key=='12487' and w.startswith('h'):reason='Mewari hiḷo cold needs the local ś/h correspondence checked.'
 if key in {'6767','7743/7785','11392'}:reason=ev
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading verification.'
 if (key,target) not in ix:ix[key,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=ev))
 i=ix[key,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=42));continue
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev+' Exact survey form '+w+' is preserved.'))
(P/'global-eighth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-eighth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_eighth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','global-eighth'))
print('accepted',len(acc),'held',len(held))
