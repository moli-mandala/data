import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
E={
'43':'CDIAL 43 akṣi explicitly gives Phal. ac̣hi, Bengali ā̃khi, Hindi ā̃kh and Nepali ā̃kho “eye”. The survey affricate/nasal/kh variants identify this eye family; contracted ā̃kā retains a phonetic qualification.',
'2485':'CDIAL 2485 ekādaśa explicitly gives Chilis/Gowro aiyāš, Palula akāš/akōš and Hindi igārah, with Western Pahari gyāra in the addendum. These variant eleven forms fit the full historical numeral compound.',
'2830':'CDIAL 2830 karṇa explicitly gives northern kan/kaṇa and phonetic koṇ, with q/k and velar-nasal notation retained in the survey. Khowar kār requires comparison with the separate *kāra ear family.',
'3084':'CDIAL 3084 kāla explicitly gives Bshk. kāl “year” and Kalasha kau, oblique kālūna. Khowar sāl needs the distinct Persian sāl loan alternative, not a mechanical substitution of the initial consonant.',
'3104-2':'CDIAL 3104.2 kalya explicitly gives Punjabi kall/kallu “yesterday, tomorrow” and the widespread kal temporal adverb. Retroflex ḷ is a local phonetic detail; initial-y yāl needs a separate correspondence check.',
'4147-3':'CDIAL 4147 distinguishes *gāvā (section 2), with Kalasha gak, Savi gāu and northern gāv, from gāvī (section 3), with Hindi gāī and Bshk. gay. Northern bare gā/gāu cannot simply be copied to gāvī without paradigm evidence.',
'4701':'CDIAL 4701 carman gives Bshk./Torwali čam and northern sām “skin”. Survey affricate ts versus c notation can identify the same skin family; added final p needs its own phonetic/lexical account.',
'5731':'CDIAL 5731 tala explicitly includes “palm, sole” and adverbial tale/tare “below”. The simple survey tal/taṛe/tel forms preserve that family, while initial aspiration or an n-bearing stem needs distinguishing alternative formations.',
'5798-4':'CDIAL 5798.1 gives Bshk. tār “star”; 5798.3 gives Kalasha tāri; 5798.4 tāraka gives Palula tōru and western tārā. The selected target is chosen by the explicit regional branch, not copied from the candidate’s masculine-star parent.',
'6251':'CDIAL 6251 dākṣiṇa explicitly gives Nepali dāinu and Hindi dāhnā/dā̃yā “right”. The d-initial variants fit; northwestern saya/sayā is not established by changing d to s and needs a separate root comparison.',
'7067':'CDIAL 7067 nāman explicitly gives Kalasha nom, northern nam and western/eastern nāũ “name”. These ordinary phonetic and case forms fit the name paradigm. No Khowar Persian-versus-Indic ambiguity is resolved by this pass.',
'8209':'CDIAL 8209 pibati gives the widespread pi-/piy- “drink” verb. Ordinary piyo/piye inflections fit; pī-lo can include the take light verb, and pī-ryo a progressive construction, so these need full-predicate analysis.',
'9201':'CDIAL 9201 *bājjara explicitly gives Punjabi bājrā, Hindi bājrā/bājṛā and Nepali bājuro “millet”. The survey bāyra/bāyarā forms preserve this distinctive grain family with local j/y development, and the deeper reconstruction remains uncertain.',
'9321':'CDIAL 9321 *bōll gives regional bol-/boḷ- “speak” and distinguishes a separate causative. The simple imperative/past bolal/bolau variants support this speak family; initial y in yolā needs a local sound account.',
'10816':'CDIAL 10816 retra explicitly gives regional ret/retā “sand”. Its addendum qualifies the weak evidence for the reconstructed tr cluster. Survey final aspiration is retained as a local phonetic uncertainty without claiming a different sand root.',
'12335':'CDIAL 12335 śarīra gives the body noun sarīra/sarīr. Forms with final r/l/j or initial h need a local correspondence account before a one-edit comparison can identify this family confidently.',
'12548':'CDIAL 12548 śuṣka explicitly gives northern śukh/šuko, Hindi sūkhā and the -ll extension behind Oriya sukhilā “dry”. This supports simple sūk/sūkhī and l-extended forms; Nepali sukeko is a verbal participle requiring the underlying dry verb to be identified.',
'13276':'CDIAL 13276 sarva gives sab/sap and the emphatic pronominal family. Dotyali sabāi fits a sab-plus-emphasis formation, while saf and sāŋ need distinct phonetic/morphological comparison.',
'13720-2':'CDIAL 13720 stoka explicitly separates the -ḍ extension giving Punjabi thoṛā and Hindi/Marwari thoṛā/thoṛo “few”. The existing *stokaḍa parent selects that extension; t/ṭ and aspiration variation remain local phonetic qualifications.',
'13992':'CDIAL 13992.1 haridrā gives Hindi harad/hardī/haldī, Kumauni haldo and Middle Indo-Aryan haraddā/haladdā “turmeric”. These simple survey metathesized or contracted forms fit the base turmeric noun, distinct from the hāridra coloured adjective.'}
current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'};acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=norm(r['Form']);target=k;reason=None
 if r['ID'] in current:continue
 if k=='2830' and r['Language_ID']=='Kho':reason='Khowar kar ear needs comparison with *kāra, expressly cross-referenced by CDIAL.'
 elif k=='3084' and r['Language_ID']=='Kho':reason='Khowar sal year requires the Persian sāl alternative rather than kāla.'
 elif k=='3104-2' and w.startswith('y'):reason='Initial y in the temporal adverb needs a local correspondence.'
 elif k=='4147-3':
  if r['Language_ID'] in {'Buksa','Chitwan'}:target=k
  elif r['Language_ID']=='Kal':target='4147-2'
  else:reason='Northern cow paradigm remains ambiguous between gāvā/gāvī or direct gām; local oblique/plural evidence is needed.'
 elif k=='4701' and w.endswith('p'):reason='Final p in the skin word requires a local phonetic or lexical account.'
 elif k=='5731' and ('n' in w or w.startswith('th')):reason='Below form needs distinguishing an n-bearing or aspirated stem from tala.'
 elif k=='5798-4':
  if r['Language_ID']=='Bshk':target='5798'
  elif r['Language_ID']=='Kal':target='5798-3'
  elif r['Language_ID'] in {'Chil','Gowro','Mai'}:reason='Bare tar star does not decide the tārā/tārakā branch without local gender/paradigm evidence.'
 elif k=='6251' and w.startswith('s'):reason='Northwestern saya right needs a separate root comparison; dākṣiṇa is not supported by this superficial match.'
 elif k=='8209' and (w in {'pilo','pil','pilu'} or 'ry' in w or w=='pis'):reason='Drink response needs past-suffix, causative or light-verb/progressive segmentation.'
 elif k=='9321' and w.startswith('y'):reason='Initial y in speak needs a local phonological account.'
 elif k=='12335':reason='Body-word consonant change needs local evidence before selecting śarīra.'
 elif k=='12548' and 'eko' in w:reason='The dry participle needs the verbal base and -eko morphology, not a direct copy to the adjective.'
 elif k=='13276':reason='The all response needs local emphasis/suffix or final-consonant analysis beyond the simple sarva family.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs local review.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=29));continue
 c=x['comparanda'][k][0];ev=E[k]+' Reviewed comparator: '+c['record']['Language_ID']+' '+c['record']['Form']+' “'+c['record']['Gloss']+'” ('+c['record']['ID']+'). Any uncertain transfer between Indo-Aryan languages remains open under the user’s preference; source phonetic details are preserved.'
 if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty concerns locality attribution rather than the lexical reading, as established by the source audit.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'near-second-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-second-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_second_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','near-second-decisions.json').replace('sixth','near-second'))
print('accepted',len(acc),'held',len(held))
from collections import Counter
print(Counter(x['parent'] for x in acc))
