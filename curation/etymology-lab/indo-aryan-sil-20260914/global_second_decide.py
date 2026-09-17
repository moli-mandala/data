import json,csv,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
E={
'2530':'CDIAL 2530 eṣa groups the proximal paradigm, with Hindi yah/is, Gujarati e, eastern e/i, and explicit Khowar hes. Survey y/j forms and proximal plurals are paradigm continuations with analogical remodeling; no direct mechanical derivation of each inflected form from nominative eṣa is asserted.',
'8082':'CDIAL 8082 pānīya documents pāni/pāṇi “water”; its addendum explicitly includes Western Pahari “water, rain”. Where a survey uses the same local form for water and rain, that recorded polysemy supports the rain response’s link to the water family.',
'10511':'CDIAL 10511 yuṣmad explains the t-initial plural/honorific pronoun through Middle Indo-Aryan tumhe with influence from tuvam. It explicitly lists Chilis/Gowro/Maiya tus and Lahnda/Punjabi tusī; western the is a contracted regional paradigm continuation, with cross-IA transfer left open.',
'986':'CDIAL 986 asmad gives the plural family amhē, regional ame/āmhi, Hindi/Bihari ham and northern mo/ma. A survey’s plural use supports this family; singular ham requires confirmation of the local singular-use convention rather than silently changing the elicited person/number.',
'12815':'CDIAL 12815 sa/so/se explicitly gives Khowar/Kalasha/Gawri se, Maiya soh and eastern se “he/that”, with feminine sā and later paradigm remodeling. Simple local deictic forms fit this family; prefixed aso requires separate segmentation.',
'14029':'CDIAL 14029 hastatala explicitly derives palm terms hathelī/thelī, with Punjabi, Hindi and Kumauni comparanda and Hindi-to-Gujarati transmission. The l-bearing survey variants retain this historical hand-plus-surface compound; bare hath or r-bearing forms require a different or locally justified analysis.',
'9051':'CDIAL 9051 phala gives eastern phol/phar and Gujarati/Marathi phaḷ “fruit”, with phali “pod” separately discussed. Fruit responses fit directly; groundnut responses may involve specialization or shortening of a modern compound and remain under review.',
'2712':'CDIAL 2712.1 kadala gives Prakrit kēla, Bengali kalā, Hindi kelā, Gujarati keḷ and Punjabi kellā “banana”. These survey l-bearing banana forms select the base branch, not the separately numbered retroflex kaḍalī.',
'7136':'CDIAL 7136 nikaṭam gives Hindi/Punjabi neṛe/nīṛe, Bshk. nīer and the addendum’s Western Pahari neḍi/neḍḍhi “near”. These local dental/retroflex-r adverbial forms are explicitly documented, including near-preserving sense.',
'13271':'CDIAL 13271 sarpa gives Prakrit sappa, Bengali sāp, Ambala samp and widespread sap “snake”. Initial h, retained r, or final aspiration need local correspondence or transmission evidence beyond an older exact link.',
'3949':'CDIAL 3949 *gakṣa/gaccha gives Bengali/Nepali gāch, Assamese gās, Oriya gacha and Bihari gāch “tree”. The survey affricate and sibilant forms match the eastern tree family; the head’s deeper formation remains reconstructed.',
'2757':'CDIAL 2757 kaphoṇi gives Prakrit kuhaṇī, Hindi kohni/kehunī, Bengali kanui and Western Pahari khunni “elbow”. The simple kohoni/kuhuni/khuni forms fit these contractions; no bāhu-containing compound is treated as a simple stem.',
'7785':'CDIAL 7785 panthā explicitly gives Khowar pon, Kalasha phon, Maiya pan and Torwali pān “path”. These northern n-retaining forms fit the head; eastern pot requires comparison with the separate patha family.',
'13139':'CDIAL 13139 sapta explicitly gives Khowar sot, Kashmiri sath, Bengali sāt and other regional sat “seven”. Retroflex final ṭ in the survey requires local review instead of treating consonant place as typographic equivalence.',
'7655':'CDIAL 7655 pañca explicitly gives Gawri ponc, Khowar ponj, Bshk. panž, Punjabi panj and eastern pā̃c/pā̃s “five”. Survey affricate/sibilant variants identify this numeral family, while an unexplained final glide is held separately.',
'12772':'CDIAL 12772 śvitra explicitly discusses Punjabi/Lahnda ciṭṭā and Jaunsari ciṭṭō “white” as a historically diffused family with debated early development. The addendum notes a citra alternative for a neighboring form; the provisional family link retains that qualification and does not settle the remote reconstruction.',
'6227':'CDIAL 6227 daśa gives Bengali das, Western Pahari doś, Bshk. daš and Marathi dahā/Konkani dhā “ten”. The survey sibilant/dental forms belong to this numeral family; the article’s explicit Hindi transmission in Punjabi remains compatible with a provisional comparative link.',
'4272':'CDIAL 4272 *goḍḍa explicitly gives Nepali goṛo “foot, leg”, Bengali goṛ, Oriya goṛa and Bihari/Hindi goṛ. These survey leg forms preserve that limb family; its connection with the wider heel/ankle/knee group remains qualified.',
'3727':'CDIAL 3727 kṣura explicitly includes the churī “knife” family through Prakrit chura/churī and notes that ch-forms spread across dialect boundaries. The survey churi/curi forms therefore receive qualified family links, without a claim of uninterrupted local inheritance.',
'3832':'CDIAL 3832 kharva documents Punjabi/Lahnda khabbā “left” and its spread, with Kalasha khāulī under an l-extension and khauṛi under another extension. Nearby kholi/khor forms can be linked provisionally to this defective-word family while the local extension and transmission remain open.'}
a=json.loads((P/'global-exact-candidates.json').read_text());rain=[x['record'] for x in a if x['parents']==['8082']];rainkeys={(r['Language_ID'],r['Tags'],norm(r['Form'])) for r in rain};water=collections.defaultdict(list)
for r in csv.DictReader((ROOT/'cldf/forms.csv').open()):
 if 'water' not in {g.strip().lower() for g in r['Gloss'].split(';')}:continue
 key=(r['Language_ID'],r['Tags'],norm(r['Form']))
 if key in rainkeys:water[key].append({k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']})
(P/'water-rain-matrix.json').write_text(json.dumps([dict(target=r,waterComparanda=water[(r['Language_ID'],r['Tags'],norm(r['Form']))]) for r in rain],ensure_ascii=False,indent=1))
qs=[dict(parent=k,citation='CDIAL['+k+']',evidence=v) for k,v in E.items()];ix={q['parent']:i for i,q in enumerate(qs)};acc=[];held=[];current={r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1'}
for x in a:
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=norm(r['Form']);reason=None;extra=''
 if r['ID'] in current:continue
 if r['ID'] in {'f_b24vhtsq6trag','f_cirlwby2um4z4','f_mdkwky6gitydq','f_i5ue2qvkhymao'}:reason='Western a/ā proximal is ambiguous between ayam and eṣa/eta families; Chatterji 1926 II pp. 832–833 explicitly distinguishes Gujarati ā from the Rajasthani paradigm. See proximal-correction.json.'
 elif k=='986' and r['Gloss']=='I':reason='The singular ham response needs local singular-pronoun evidence.'
 elif k=='12815' and w.startswith('a'):reason='Initial a in aso requires segmentation/paradigm review.'
 elif k=='14029' and (w in {'hath','hat'} or 'r' in w or 'ṛ' in w):reason='Bare hand noun or r-bearing palm form needs distinct analysis/local l-to-r evidence.'
 elif k=='9051' and (r['Gloss']=='groundnut' or w=='phav'):reason='Groundnut specialization/compound shortening or final v needs separate review.'
 elif k=='2712' and w.startswith('kh'):reason='Initial aspiration in this banana form requires local confirmation.'
 elif k=='13271' and (w.startswith('h') or 'r' in w or w.endswith('ph')):reason='Initial h, retained r or final aspiration needs local evidence for this snake form.'
 elif k=='7785' and r['Language_ID']=='Hajong':reason='Eastern pot may belong to patha rather than panthā.'
 elif k=='13139' and 'ṭ' in w:reason='Retroflex final stop in seven needs local correspondence/source review.'
 elif k=='7655' and w=='pay':reason='Final glide in five needs a local account rather than exact-link copying.'
 if k=='8082':
  ws=water[(r['Language_ID'],r['Tags'],w)]
  if 'rain' in r['Gloss'] and not ws:reason='Rain usage needs a same-locality water comparator or direct local lexical evidence.'
  elif ws:extra=' Same language, locality tags and normalized form are recorded for water: '+', '.join(z['ID'] for z in ws[:3])+'.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires local review.'
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=19));continue
 ev=E[k]+extra+' The exact regional route may include borrowing between Indo-Aryan languages; the supported family is linked under the user’s preference.'
 if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty is source-locality attribution, not lexical reading; tags are retained.'
 acc.append(dict(record=r,family=i,parent=k,citation=qs[i]['citation'],evidence=ev))
(P/'global-second-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-second-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_second_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','global-second-decisions.json').replace('sixth','global-second'));print('accepted',len(acc),'held',len(held))
