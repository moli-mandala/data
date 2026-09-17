import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
E={
'10066':'CDIAL 10066 mārayati gives the mār- kill verb; maral/marla and māryo are ordinary eastern -l and western -y past/infinitive formations. Mār-diyo contains a give light verb and is not assigned as a simple inherited stem.',
'9871':'CDIAL 9871 marate gives Punjabi marnā and Marwari marṇo “die”, but mar-gyo and mar-go contain a go light verb whose complete construction needs analysis.',
'12598':'CDIAL 12598 śṛṇoti explicitly gives Bhojpuri sunal and western suṇṇo “hear”. Eastern -l and western -y inflections can continue the hear stem; sun-lo and ambiguous sunā causative/past forms require local morphology.',
'7655':'CDIAL 7655 pañca explicitly gives Dardic panž/panz and Pashai paĩ/paẽ, alongside Kalasha poñ and Shina poĩ. The survey affricate-to-sibilant and contracted nasal/glide numerals fit this documented five family; exact local phonetic developments remain qualified.',
'2575':'CDIAL 2575 kaḥ punar gives kaun/koṇ/kuṇa “who” and Old Gujarati kuṇaïṃ instrumental. The q/k variants and final pronominal vowels preserve the family; Kalasha kura needs comparison with a different interrogative stem.',
'5086':'CDIAL 5086 jaṭā gives Bi. jar and Nepali jari/jaro “root”. Nasal-initial nar may be the separate nāḍī family; palatal/aspirated initial deviations need comparison before assigning.',
'2668-2':'CDIAL 2668.2 kaṇṭaka explicitly gives Punjabi/Lahnda kaṇḍā, Palula kāṇḍu and Western Pahari kā̃ḍā “thorn”, with Garhwali kā̃ḍu in the addendum. These ending-bearing forms select kaṇṭaka, not bare kaṇṭa.',
'7733':'CDIAL 7733 pattra explicitly gives Lahnda pattar, Punjabi pattā and Bshk. paλ “leaf”. The l/ɬ notation in the latter and regional r-bearing or simple pata forms fit this branch; pana/pan may instead be parṇa.',
'6507-2':'CDIAL 6507.2 *dekṣati gives the dekh- see verb and regional v-/d- variation through transmission. Dekh-ta, dekh-u and ordinary -y past forms retain that stem; dekh-liyo requires the take light verb to be analyzed.',
'10978':'CDIAL 10978 lavaṇa gives the widespread loṇ/lūṇ salt family. Regional retroflex initial ḷ and the northern retained-v lovn do not require a different lexical root; their phonetic details remain in the source transcription.',
'6152':'CDIAL 6152 danta explicitly gives Maiya/Palula dān, Gawri dant and northwestern dand “tooth”; its addendum includes aspirated Sindhi dandh. The simple dān/dant/dandh survey forms therefore fit this family.',
'10234':'CDIAL 10234 mūtra explicitly gives Kalasha mūtra, Bshk. mūλ and Western Pahari mūc. The survey lateral/sibilant outcomes fit the tr-cluster urine family; retroflex muṭ and ordinary mute preserve the same noun, with phonetic details left open.',
'4655-2':'CDIAL 4655.3 caturaḥ explicitly gives Bshk. čōr, Kalasha čāu, Khowar čhor, Maiya čōur and Palula čūr. These northern o/u forms require subsection 3, correcting the candidate’s subsection-2 link; western cār remains in subsection 2.',
'3083':'CDIAL 3083 kāla explicitly gives Bihari kariyā and Sindhi kāro “black”, with Western Pahari kāo. Survey ordinary vowel/gender variants retain this black adjective; karḍo/kaḍo needs separate extension analysis.',
'135':'CDIAL 135.1 aṅguli explicitly gives Maithili ā̃gur, Nepali aũlo and Lahnda aṅgal “finger”. The Maiya agui-type form is separately listed under 135.2 *aṅgūḍi and is assigned there; opaque reduced or retroflex-r forms remain distinct audit questions.',
'9349':'CDIAL 9349 bhaginī gives Bshk-area bai(n), Kashmiri biñ, Punjabi bhaiṇ and regional bahin/bheṇ “sister”. Simple n-retaining variants can be linked provisionally; additional r in baharṇ requires a separate morphological account.',
'986':'CDIAL 986 asmad explicitly gives the ham/amh- and northwestern asī/asā we paradigms. Qualified inclusive/exclusive elicitation labels are preserved, without inventing a different root where the same documented form is used.',
'4721':'CDIAL 4721 calati gives the cal-/cāl- walk verb. The selected calā/cāllo forms are ordinary verbal inflections; initial h or zero needs local correspondence evidence.',
'8857':'CDIAL 8857 prastara explicitly gives Prakrit patthara and regional pathar/pathara “stone”, including known cross-IA loans. Unaspirated patar remains a qualified phonetic variant within this identifiable stone family.',
'7136':'CDIAL 7136 nikaṭam explicitly gives Bshk. nīer and northwestern neṛe/neḍi/neḍḍhi “near”. The survey nīar/niḍe forms fit; initial m in mere remains a different comparison.',
'9758':'CDIAL 9758 matsya explicitly gives Bshk. fish plural -cin and the eastern macharī -l/-r extension. Survey maśīn and macharī match those documented paradigmatic/extended forms, not the separate matsiya branch.',
'2757':'CDIAL 2757 kaphoṇi gives Hindi kohni/kehunī, Bengali kanui and Western Pahari khunni “elbow”. Simple contracted q/k and vowel variants preserve this historical elbow family.',
'10158':'CDIAL 10158 mukha explicitly gives mouth/face mu/muh and retained-kh mukha forms. The simple survey moho/māī and mukh responses preserve these attested contractions or learned/regional forms.',
'6495':'CDIAL 6495 dūra gives Palula dhūro and regional dūr “far”, with retroflex ḍūr in the addendum. The selected survey variants preserve the distance adjective; local r/ṛ articulation does not by itself establish another root.',
'8265':'CDIAL 8265 putra explicitly gives northwestern puttar/puttur and Kashmiri pothar “son”. The r-retaining and ordinary vowel variants fit this family, with transmission and aspiration details retained.',
'5550':'CDIAL 5550 ḍimba explicitly gives Bengali ḍim and Assamese ḍimā “egg”.',
'5301':'CDIAL 5301 jyotsnā explicitly gives Nepali jun and Assamese zon “moon”.'}
current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'};acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=norm(r['Form']);target=k;reason=None
 if r['ID'] in current:continue
 if k=='10066' and ('dy' in w or 'di' in w or 'jy' in w):reason='The kill response appears to contain dī/diyo give as a light verb; retain the full compound for analysis.'
 elif k=='9871' and w not in {'marno','maṛ'}:reason='The die response has additional go/progressive material; analyze the whole predicate before assigning.'
 elif k=='12598' and (w.startswith('h') or w.endswith('lo') or w in {'suna','suni','sunau'}):reason='The hear response needs local past/causative/light-verb or initial-h analysis.'
 elif k=='2575' and ('r' in w or w.startswith('kh')):reason='Interrogative stem or initial aspiration needs a local paradigm check.'
 elif k=='5086' and (w.startswith(('n','y','jh'))):reason='Root noun may involve nāḍī or an unverified initial change; a same-gloss edit-distance match is not enough.'
 elif k=='7733' and (w in {'pan','pana'} or 'ṭ' in w):reason='Leaf form requires distinguishing parṇa or the retroflex pattra/paṭṭa history.'
 elif k=='6507-2' and ('l' in w):reason='The see response includes -l material needing past-suffix versus take light-verb analysis.'
 elif k=='3083' and 'ḍ' in w:reason='The black adjective has an additional retroflex extension needing local morphological analysis.'
 elif k=='135':
  if r['Language_ID']=='Mai':target='135-2x'
  elif w in {'aŋgi','aŋgṛi'}:reason='Reduced or retroflex-r finger form needs distinguishing aṅguli from *aṅgūḍi.'
 elif k=='9349' and 'r' in w:reason='Extra r in the sister noun requires a local historical account.'
 elif k=='986' and w=='hav':reason='Initial/final remodeling in hav we needs local paradigm support.'
 elif k=='4721' and not w.startswith('c'):reason='Initial h/zero walk form needs a local sound-correspondence account.'
 elif k=='7136' and w.startswith('m'):reason='Initial m near form is not established as nikaṭam by the candidate resemblance.'
 if k=='4655-2' and r['Language_ID'] in {'Bshk','Chil','Kal','Kho','Mai','Phal'}:target='4655-3'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires local review.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.removesuffix('x').replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=26));continue
 c=x['comparanda'][k][0];ev=E[k]+' Reviewed comparison: '+c['record']['Language_ID']+' '+c['record']['Form']+' “'+c['record']['Gloss']+'” ('+c['record']['ID']+'). The family link leaves local phonetic details and possible transfer between Indo-Aryan languages open; no source spelling is changed.'
 if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty concerns locality attribution rather than lexical reading, as established by the source audit.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'near-first-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-first-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_first_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','near-first-decisions.json').replace('sixth','near-first'))
print('accepted',len(acc),'held',len(held))
from collections import Counter
print(Counter(x['parent'] for x in acc))
