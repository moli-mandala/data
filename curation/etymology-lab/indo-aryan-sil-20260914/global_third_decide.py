import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
E={
'2574':'CDIAL 2574 ka explicitly gives eastern ke “who”, Palula ko and Khowar ka, with what/why ka in the Western Pahari addendum. Survey what/where forms need to be separated from local kim-derived or adverbial constructions; the who forms are the immediate supported subset.',
'10924':'CDIAL 10924 leaves many larika/laṛka forms between *laḍikka and *laḍḍikka with shortening. The older exact links do not settle that reconstruction/subsection choice.',
'10158':'CDIAL 10158 mukha gives Prakrit muha “mouth, face” and northern/eastern mu/muh/mui, with Khowar mux. The survey contracted mu/moh/mui forms fit the mouth/face family; unaspirated muk needs separate review of the retained stop.',
'9229':'CDIAL 9229 bāhu explicitly gives eastern bāh/bāhā and northern bā/bāi “arm”. It separately marks Khowar bāzū as Iranian or Nuristani-mediated, so that distinct donor issue is retained rather than bypassed.',
'4368':'CDIAL 4368 grāma explicitly gives Lahnda/Punjabi girā̃/grā̃ and eastern gā̃/gām, with learned gram readily identifiable. It also lists a -ṭa extension in gāmaḍa; unexplained survey gauḍa/gāvḍo contraction needs a local account.',
'3865':'CDIAL 3865 khādati gives the widespread khā- eat/bite verb and Palula kha-, with Bhojpuri khāil in the base-verb discussion. Imperative/infinitive morphology is distinct from an attached light verb le; opaque l-bearing survey responses are held for segmentation.',
'6507-2':'CDIAL 6507.2 *dekṣati gives the dekh- “see” family, including Hindi dekhna and Rajasthani dekhṇo, and explicitly discusses its spread across Indo-Aryan. Survey dekh/dekho imperatives and dekhyo past forms are ordinary inflections of that stem; no light verb is present.',
'11348':'CDIAL 11348.1 *varta explicitly gives Kalasha bat, Bshk. baṭ, Torwali bāṭ and northern baṭh/baṭṭh “stone”, with aspiration noted as unexplained in the source itself. The round-stone family also includes Punjabi vaṭṭā/baṭṭā; unexplained initial bh and a pestle specialization remain separate audit questions.',
'10702':'CDIAL 10702 rātrī gives Palula rōt, widespread rāt and the addendum’s Western Pahari rāc “night”. Thus the Kului affricate is directly supported in the neighboring primary comparison. Retroflex final ṭ remains a transcription/local-correspondence question.',
'8376':'CDIAL 8376.1 *peṭṭa gives Hindi/Bengali peṭ and Oriya peṭa “belly”, distinct from poṭṭa and peḍḍuka. The survey pet spellings preserve the diagnostic e-vowel belly family, with the source’s dental-versus-retroflex transcription retained for audit rather than silently normalized in the data.',
'3167':'CDIAL 3167 *kiyatta explicitly includes kittaka/ketek, Hindi kitnā and regional kitrā “how much/many”. The survey katek/katka/kitte/katra variants belong to these interrogative extensions; unusual Ushoji kacāk requires a local consonant account.',
'6328':'CDIAL 6328 dina means “day, daytime”. Bare dan/din day forms fit this family, while sun-only usage needs a local day comparator and dinmān has additional compound material.',
'3244':'CDIAL 3244 kuṭhāra explicitly gives kuhāḍa/kuhāṛa, Hindi kulhāṛā, Gujarati kuvāṛī and Marathi kurhāḍ “axe”. The survey forms display the same documented metathesis and l/r alternations, with cross-IA transmission left open.',
'12583':'CDIAL 12583 śṛṅga gives the s/ś/ṣ-bearing horn family, including northern ṣīṅ and Punjabi siṅg. Initial h and additional -ḍa require local correspondence or suffix evidence, not merely an older link.',
'6658':'CDIAL 6658.1 dvādaśa gives northern bāra/bāh and eastern bāra “twelve”. It explicitly separates Torwali/Maiya duāš under 6658.2 duvādaśa; those survey forms are assigned to the latter node.',
'13551':'CDIAL 13551.1 sūcī gives Hindi/Bihari sūī and the k-extension-free needle family. Initial h or medial j in these other survey forms requires local correspondence evidence; Magahi sūīyā is a transparent extended needle noun.',
'6984':'CDIAL 6984 nava “nine” lists eastern na, northern nau/nao and western nao/nau. The survey numeral meanings distinguish them from the nava “new” homonym despite similar forms.',
'14024':'CDIAL 14024 hasta explicitly gives Maiya hā and regional hath/ath “hand, arm”, including initial-h loss in Kashmiri and Romani. Survey retroflex stop spelling is preserved as an audit qualification; the hand/arm family remains well identified.',
'5958':'CDIAL 5958 taila gives Prakrit tēla/tella and regional tel “oil”, with oil/fat polysemy explicitly recorded in Romani. The survey tel/tal material-fat/oil forms identify this family; retroflex ṭ in some survey transcriptions is retained as an unresolved local phonetic detail.',
'9153':'CDIAL 9153 barkara explicitly gives Hindi bakrī, Himalayan bākhro and bokrā under contamination with *bokka. Western bokaḍī requires distinguishing that expressive contamination and its retroflex extension; simple bokari/bakheri forms fit the named goat family.',
'3197':'CDIAL 3197 kīdṛśa explicitly includes Hindi kaisā and Old Marwari kiso “of what kind”, with interrogative remodeling through ka. Survey kaiso/keso/kaso forms fit this family; final -k in kasyok is a separate morphological question.',
'11515':'CDIAL 11515 vānara explicitly gives eastern bandara/bāndar and Nepali bā̃dar “monkey”, alongside regional transfer arrows. Survey bandor/bandara forms fit this family, with the borrowing route left open.',
'6914':'CDIAL 6914.1 nakha explicitly gives Palula nāṅg and regional nahu/nauh “fingernail”. Retained-k nok forms may require the separately numbered *nakkha or learned/regional nakha transmission, so are not assigned by an old unsplit link alone.',
'11503':'CDIAL 11503.1 vātiṅgaṇa gives Punjabi/Hindi baigan and Bengali begun “eggplant”. It separately lists Bastar bāṅgā in 11503.5 vaṅga; the survey banga form selects that contracted branch, not the full head.',
'11165':'CDIAL 11165 lohita explicitly gives regional lahū/lehū/lau, Jaunsari loī and Khowar lei in the addendum. These blood forms fit the documented red-to-blood family; very contracted Khowar le remains a comparison with a possible lohī alternative.'}
known={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['ID'] in {'6658-2','11503-5'}};assert len(known)==2
qs=[];ix={};acc=[];held=[];current={r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1'}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=norm(r['Form']);target=k;reason=None
 if r['ID'] in current:continue
 if k=='2574' and not r['Gloss'].startswith('who'):reason='What/where forms require a local kim or adverbial analysis, not wholesale transfer of the who root.'
 elif k=='10924':reason=E[k]
 elif k=='10158' and w=='muk':reason='Retained k needs explicit learned/retentive versus different branch analysis.'
 elif k=='9229' and 'z' in w:reason='Khowar bazu has Iranian/Nuristani donor alternatives explicitly identified by CDIAL.'
 elif k=='4368' and any(c in w for c in ['ḍ','ṇ','ṭ']):reason='The village form needs its additional consonantal material analyzed.'
 elif k=='3865' and ('l' in w or w=='khe'):reason='An l-bearing eat response may include le as a light verb; local morphology must be separated.'
 elif k=='11348' and (w.startswith('bh') or r['Gloss']=='pestle'):reason='Initial bh or pestle specialization requires local evidence beyond the stone comparison.'
 elif k=='10702' and ('ṭ' in w or w.startswith('ṛ')):reason='Night form has an unresolved consonant-place/initial correspondence.'
 elif k=='3167' and r['Language_ID']=='Ush':reason='The affricate in this how-many form requires a local account.'
 elif k=='6328' and (r['Gloss']!='day' or w=='dinman'):reason='Sun/day polysemy or dinman compound needs separate local review.'
 elif k=='12583' and (w.startswith('h') or 'ḍ' in w):reason='Initial h or additional retroflex extension requires local support.'
 elif k=='6658' and w.startswith('du'):target='6658-2'
 elif k=='13551' and (w.startswith('h') or 'j' in w or w=='siu'):reason='Need local initial-h, medial-j, or vowel-order correspondence for the needle form.'
 elif k=='9153' and 'ḍ' in w:reason='Retroflex bokaḍi requires distinguishing *bokka contamination/extension from barkara.'
 elif k=='3197' and w.endswith('k'):reason='Final k in kasyok requires morphological analysis.'
 elif k=='6914' and ('k' in w or w=='nog'):reason='Retained k/g leaves nakha versus nakkha/transmission unresolved.'
 elif k=='11503' and w=='baŋga':target='11503-5'
 elif k=='11503' and w=='bintak':reason='Bintak has additional consonant changes not settled by a broad eggplant-family resemblance.'
 elif k=='11165' and r['Language_ID']=='Kho':reason='Khowar contracted le may also continue lohī; inspect the competing root before choosing.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs local review.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=20));continue
 ev=E[k]+' The comparative family link retains uncertainty about transfer between Indo-Aryan languages.'
 if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty is locality attribution only, as established by the source audit; source tags are retained.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'global-third-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-third-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_third_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','global-third-decisions.json').replace('sixth','global-third'));print('accepted',len(acc),'held',len(held))
