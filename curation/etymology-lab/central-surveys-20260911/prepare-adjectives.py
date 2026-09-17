from research_helpers import Batch
b=Batch(7);a=b.add
for l,w in [('Malvi','junno|juna|juṇa|juno|junna|junni'),('Nimadi','junu|juno|juṇo')]:
 a(l,'old',w,'5260','CDIAL 5260 gives Prakrit juṇṇa, Old Awadhi jūna, Hindi jūnā and Gujarati jūnũ ‘old’. The assimilated rṇ cluster, with singleton/geminate and gender-ending variation, fits these forms.')
a('Bagheli','old','juneha','5260','The jūn- stem matches Prakrit juṇṇa and Hindi jūnā (CDIAL 5260). The extended -eha ending remains morphologically unresolved, so only the historical family is proposed for triage.',tier='qualified')
for l,w in [('Malvi','puraṇo|purānā|purāṇo|purano'),('Nimadi','purāṇu|puraṇo|purāno'),('Bagheli','purana|puran|puṛana|puṛaṇa')]:
 a(l,'old',w,'8283','CDIAL 8283 compares Prakrit purāṇa, Hindi purānā, Bhojpuri purān and Gujarati purāṇũ. These support the old adjective with regional final endings; the source’s r/ṛ and n/ṇ distinctions are retained.',tier='qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','navo|nava|navi|navā'),('Nimadi','navo|nāvo'),('Bagheli','neba|neua|neuo|nau')]:
 a(l,'new',w,'6983','CDIAL 6983 gives Prakrit ṇa(v)a, Old Awadhi nava, Marwari navo and Gujarati navũ. '+('Bagheli e-vowels and b/w loss require a dialectal check; návya is an alternative for contracted forms.' if l=='Bagheli' else 'The retained v distinguishes these from the y-bearing naviya family.'),tier='qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','nəyā|nayo|naya'),('Nimadi','naya|nayo'),('Bagheli','neyə')]:
 a(l,'new',w,'7025-2','CDIAL 7025.2 specifically gives naviya, Prakrit ṇavia, Hindi/Bihari nayā and Old Marwari nayaü. The y-bearing stem fits this subsection rather than the unsplit navya head.',tier='qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','acho|acːo|āchā|achā|āchi'),('Nimadi','acho|acco|accho|ācho'),('Bagheli','eca|ece|echa')]:
 a(l,'good',w,'142','CDIAL 142 compares Prakrit accha ‘clear, pure’, Old Marwari āchyo/āchī and Punjabi acchā ‘good’, and explicitly derives standard Hindi acchā through Punjabi. The family is plausible, but local inheritance versus a Hindi/Punjabi-mediated loan is unresolved; no immediate donor is asserted.',tier='qualified',locator='142.1')
for l,w in [('Malvi','suko|sukːo|sukho|suka|sukhā|śukːā|sukha'),('Nimadi','sukho'),('Bagheli','suka|sukh|sukha')]:
 a(l,'dry',w,'12548','CDIAL 12548 compares Prakrit sukkha/sukka, Hindi sūkhā/sūkā, Gujarati sūkũ and Marathi sukā. Assimilation of ṣk and the attested aspiration variation account for this dry-adjective group.')
a('Malvi','dry','hukːo|hukho|hukːā|huko|hukːa','12548','The medial kk/kh and dry sense match Prakrit sukkha/sukka and western sūk- forms in CDIAL 12548. Initial h requires confirmation of a local s-to-h correspondence; Kashmir’s similar h is not evidence of a direct donor.',tier='qualified')
for l,w in [('Malvi','lambo|lambā|lāmba'),('Nimadi','lambo|lāmbo')]:
 a(l,'long',w,'10951','CDIAL 10951 gives Prakrit laṁba, Hindi lambā/lā̃b, Old Marwari lāṁbaü and Gujarati lā̃bũ ‘long’. The labial cluster and gender endings match directly.')
a('Bagheli','long','lemba|lembay|lembi|lemmi','10951','CDIAL 10951 compares Hindi lambā and Punjabi lambā/lammā, illustrating both retained and assimilated mb. Bagheli e and the final -ay require local confirmation; the source spellings remain unchanged.',tier='qualified')
for l,w in [('Malvi','tāto|tato|tatā'),('Nimadi','tāto')]:
 a(l,'hot',w,'5679','CDIAL 5679 traces hot adjectives through Pali/Prakrit tatta, with Hindi tātā, Marwari tāto and Gujarati tātũ. The pt > tt > t development provides direct support.')
a('Bagheli','hot','ṭaṭ|ṭaṭh','5679','Compare Awadhi tāt and Hindi tāt/tātā in CDIAL 5679. The source’s two retroflex stops and occasional final aspiration are not explained by the cited comparanda, so the proposed family needs phonological review.',tier='qualified')
for l,w in [('Malvi','ṭhanḍo|ṭaṇḍo|ṭanḍo|ṭhaṇḍo|ṭhaṇḍa|ṭaṇḍā'),('Nimadi','ṭhaṇḍo|ṭhanḍo|tānḍo|ṭaṇḍo'),('Bagheli','ṭeṇḍa|theṇḍ|theṇḍa')]:
 a(l,'cold',w,'13676','CDIAL 13676.2 derives the cold family from stabdha via ṭhaḍḍha, but treats nasal *thaṇḍha/*ṭhaṇḍha as influenced by Dravidian cold words and documents Hindi loans into several languages. The family link is qualified: contact contribution and the survey’s immediate transmission remain unresolved.',tier='qualified',locator='13676.2')
for l,w in [('Malvi','dur|durā'),('Nimadi','dur'),('Bagheli','ḍuri')]:
 a(l,'far',w,'6495','CDIAL 6495 gives Prakrit dūra, Hindi/Bhojpuri dūr, Old Awadhi dūri and Old Marwari durī. '+('Bagheli’s initial ḍ is preserved and needs the survey’s dental/retroflex correspondence checked.' if l=='Bagheli' else 'Both bare and vowel-final distance adverbs have direct regional comparanda.'),tier='qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','moṭo|moṭā|moṭa'),('Nimadi','moṭo')]:
 a(l,'big',w,'10187-11','CDIAL 10187.11 *mōṭṭa specifically compares Old Marwari moṭaü ‘big, fat’, Gujarati moṭũ and Marathi moṭā. This selects the relevant eleventh branch rather than root *muṭṭa ‘defective’.')
for l,w in [('Malvi','coṭa'),('Nimadi','choṭo|coṭo'),('Bagheli','choṭe|coṭa')]:
 a(l,'small',w,'5071','CDIAL 5071 gives Hindi choṭā, Awadhi choṭ and Gujarati choṭũ ‘small’. The addenda distinguish related *chōṭa/*cōṭa forms, so the deaspirated variants are grouped as a qualified small-word family rather than a proven unique reconstruction.',tier='qualified')
for l,w in [('Malvi','bhari|bhāri'),('Nimadi','bhāri'),('Bagheli','bhaṛi')]:
 a(l,'heavy',w,'9465','CDIAL 9465 bhārika ‘heavy’ gives Bihari bhāri and Awadhi/Hindi/Gujarati/Marathi bhārī. The adjective belongs to this specific formation rather than the bare noun bhāra ‘load’.',tier='qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','haḷəko|halko|halkā|halka'),('Nimadi','haḷoko|haḷko|hālko|halko'),('Bagheli','helka|heluk|halka|haluk')]:
 a(l,'light',w,'10896-5','CDIAL 10896’s -kk- extension with metathesis compares Hindi halukā/halkā, Bihari haluk and Marathi haḷkā. It explicitly records Hindi loans into Gujarati and Marathi, so the extended family is proposed while local inheritance versus contact remains open.',tier='qualified',locator='10896, -kk- extension with metathesis')
for l,w in [('Malvi','kaḷo|kāḷo|karo|kāḷā|kala|kalo|kaḷa|kālā'),('Nimadi','kāḷo|kāḷā'),('Bagheli','kala|keriya')]:
 a(l,'black',w,'3083','CDIAL 3083 compares Prakrit kāla, Marwari kāḷo, Gujarati kāḷũ, Hindi kālā and Bihari/Bhojpuri kariyā. The l/r forms are represented in the family; this proposal does not claim an Indo-European origin for kāla itself.',tier='qualified' if l=='Bagheli' else 'straightforward',locator='3083.1')
for l,w in [('Malvi','dhoḷo|dhoḷā|dhoro|dhoḷa'),('Nimadi','dhauḷo|dhāvḷo')]:
 a(l,'white',w,'6767','CDIAL 6767 compares Old Gujarati dhaülaü, Gujarati dhɔḷũ, Hindi dhaulā/dhorā and Marathi dhavaḷ. Both contracted and v-bearing forms fit the dhavala ‘white’ family.')
b.save()
