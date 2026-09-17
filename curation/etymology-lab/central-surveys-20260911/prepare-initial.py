"""Materialise explicitly researched decisions, never accepted overlay rows."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
parents=json.loads((p/'parents.json').read_text())
inv={l:json.loads((p/f'{l}-inventory.json').read_text()) for l in ['Malvi','Nimadi','Bagheli']}
props={l:[] for l in inv};used=set()
def add(lang,gloss,words,parent,evidence,tier='straightforward',locator=None,kind='reflex'):
 words=words.split('|');rows=[r for r in inv[lang] if r['Gloss']==gloss and r['Form'] in words]
 assert rows,(lang,gloss,words)
 assert not used.intersection(r['ID'] for r in rows),(lang,words)
 assert parents[parent] is not None,parent
 used.update(r['ID'] for r in rows)
 n=len(props[lang])+1;cite='CDIAL['+(locator or parent.replace('-', '.',1))+']'
 proposal={'number':n,'status':'pending-review','difficulty':tier,'formIds':[r['ID'] for r in rows],'forms':list(dict.fromkeys(r['Form'] for r in rows)),'gloss':gloss,'records':rows,'parentId':parent,'parentForm':parents[parent]['word'],'kind':kind,'citation':cite,'evidence':evidence,'assignments':[{'Form_ID':r['ID'],'Etymon_ID':parent,'Kind':kind,'Rank':'1','Status':'accepted','Source':cite,'Notes':evidence+' Pending central-survey proposal '+lang+' '+str(n)+'.','Pos':''} for r in rows]}
 props[lang].append(proposal)
# Full CDIAL entries and their addenda read in cdial-articles.json; no automatic matching claims.
for l,words in [('Malvi','mātho|māthā|matha'),('Nimadi','mātha|mātho|māthā')]:
 add(l,'head',words,'9926','Pali mattha/matthaka and Prakrit mattha lead to Hindi māthā and explicitly Marwari mātho ‘head, forehead’. The dental aspirate and vowel length fit the first branch of masta/mastaka; no brain-related subsection is intended.',locator='9926.1')
for l,words in [('Malvi','madho|madhā|mādā|mādo|mato|matā'),('Nimadi','mādho|madho')]:
 add(l,'head',words,'9926','Compare the same survey’s māthā/mātho and CDIAL’s Marwari mātho, Gujarati māthũ. Voicing or loss of aspiration differentiates these local forms; the family is plausible, but these developments need locality-specific confirmation.','qualified',locator='9926.1')
for l,words in [('Malvi','bal|bāl|baḷ'),('Nimadi','bāl'),('Bagheli','bal|bar')]:
 add(l,'hair',words,'11572','CDIAL gives Hindi bāl, Gujarati vāḷ and Bhojpuri bār ‘hair’ beneath vāla. Initial v > b and, in Bagheli bar, the regional l/r correspondence have explicit comparanda.')
for l,words in [('Malvi','kān|kāṇ'),('Nimadi','kān'),('Bagheli','kan')]:
 add(l,'ear',words,'2830','Prakrit kaṇṇa corresponds to Hindi, Old Marwari, Gujarati and Marathi kān ‘ear’ in CDIAL. The local dental/retroflex nasal and variable written vowel length fit this inherited family.')
for l,words in [('Malvi','dāt'),('Nimadi','dāt|dā̃t|dānt'),('Bagheli','dat|dāt')]:
 add(l,'tooth',words,'6152','CDIAL explicitly gives Hindi, Marwari, Gujarati and Marathi dā̃t from danta. Reduction of nt with vowel lengthening/nasalisation accounts for these forms; survey notation varies in whether nasalisation is written.')
for l,words in [('Malvi','jib'),('Nimadi','jib|jibə|jip'),('Bagheli','jib')]:
 add(l,'tongue',words,'5228','Prakrit jibbhā and Hindi jīb(h), Gujarati jībh and Marathi jībh are cited under jihvā. The reduced labial cluster fits jib; Nimadi jip additionally shows final devoicing.',locator='5228.1')
for l,words in [('Malvi','peṭ|peṭh|piṭ'),('Nimadi','peṭ|peṭh'),('Bagheli','peṭ|peṭe|pēṭ')]:
 add(l,'belly',words,'8376','CDIAL’s first branch has Prakrit peṭṭa/piṭṭa and Hindi peṭ, Old Marwari peṭa, Gujarati peṭ; Maithili peṭ(h) also supplies an aspirated comparator. This is the e/i-vowel branch, separate from pōṭṭa.',locator='8376.1')
add('Malvi','belly','poṭ','8376-3','CDIAL’s third branch explicitly gives Prakrit poṭṭa/puṭṭa and Marathi poṭ ‘belly’. The o vowel motivates this specific subsection rather than the peṭṭa head.',locator='8376.3')
for l,words in [('Malvi','uŋgəḷi|aŋgḷi|āŋgḷi|aŋgḷiya|aŋgaḷi|uŋgḷi|aŋgri|aŋgiḷi|aŋgəḷi|uŋgili|aŋgali'),('Nimadi','aŋgḷai|aŋgəḷi|angəlai|āŋgḷi|āŋgḷyā|aŋgḷei'),('Bagheli','uŋgli|eŋguṛi|eŋguli|aŋguli|uŋgali|uŋgiriya')]:
 add(l,'finger',words,'135','CDIAL lists Hindi uṅglī, Gujarati ā̃gḷī, Marathi ãgḷī and Old Hindi ā̃gurī under aṅguli/aṅguri. Syncope and l/ḷ/r variation support this branch; survey-specific extended endings are retained in the proposal.','qualified',locator='135.1')
for l,words in [('Malvi','nakh|nukh'),('Nimadi','nakh|nākh'),('Bagheli','nekh')]:
 add(l,'fingernail',words,'6914-2','The retained kh is compared with CDIAL’s explicitly strengthened *nakkha branch (Prakrit ṇakkha, Bshk. nakh, Torwali nōkh), rather than the nah-/noh- reflexes of plain nakha. A learned or contact-supported retention cannot yet be excluded in these central varieties.','qualified',locator='6914.2')
add('Bagheli','fingernail','neh|neha|nah|nāha|nəhe|ne','6914','CDIAL’s first branch lists Prakrit ṇaha, Maithili nah/nauh, Bhojpuri nõh and Hindi nahã. Loss of the intervocalic velar and local vowel variation place these with that branch, unlike nekh or the loan nākhun.','qualified',locator='6914.1')
for l,words in [('Malvi','caməḍo|caməḍi|camaḍa|camḍa|cāmaḍo|camḍi|caməda|cāmḍi|cāmaḍa'),('Nimadi','cāməḍi|caməḍo|cāmbḍo|cəməḍā|cāmḍā|camḍe|cāmaḍi|cāmaḍo'),('Bagheli','cemeḍe|cemaḍi|cemeḍi|cemeṛi|cemṛa|cemṛe|cemṛi|cam')]:
 add(l,'skin',words,'4701','CDIAL gives Hindi camṛā, Gujarati cāmḍũ/cāmḍī and Marathi cāmḍẽ/cāmḍī as the historical -ḍ- extension of carman via camma. These survey forms fit that inherited extended family; cāmbḍo additionally has an intrusive b, so it remains qualified.','qualified')
for l,words in [('Malvi','haḍḍā|haḍḍi|haḍ'),('Nimadi','haḍḍi|hāḍḍi|haḍḍa'),('Bagheli','heḍe|heḍḍe|haḍ|haḍi|haḍə|hāḍā|hāṛ')]:
 add(l,'bone',words,'13952','Prakrit haḍḍa, Hindi haḍḍā/haḍḍī and hāṛ, and Gujarati/Marathi hāḍ give direct comparanda. Turner explicitly calls a connection to asthi very doubtful; the proposal stops at haḍḍa.')
for l,words in [('Malvi','haḍəka|haḍikā|haḍiko|haḍəko|haḍəki|haḍiki|haḍikyo|haḍikyā|haḍikya|haḍika'),('Nimadi','haḍəka|haḍəkā|haḍəko|hadki|hāḍkā')]:
 add(l,'bone',words,'13952','The base agrees with haḍḍa; CDIAL’s addenda explicitly provide k-extended haṛkɔ, hāḍkī and Garhwali hāḍgu (< *hāḍku?). The survey k-forms fit this extended family, with the exact age/productivity of the suffix left open.','qualified')
for l,words in [('Malvi','gam|gām|gā̃v|gāv|gā̃'),('Nimadi','gāũ|gā̃v|gāv'),('Bagheli','gə̃u|gāũ|gau|gāu')]:
 add(l,'village',words,'4368','CDIAL lists Prakrit gāma, Gujarati gām, Marwari gā̃v and Hindi gā̃u beneath grāma. The nasal and v/u outcomes are directly represented, without assigning the separate dehāt or khelo responses.')
for l,words in [('Malvi','kuni|koni|kohni'),('Nimadi','koṇi|koiṇi|koini|kuṇi|kohini'),('Bagheli','kehuni|kihuni|koheni|kuhuni')]:
 add(l,'elbow',words,'2757','CDIAL gives Prakrit kuhaṇī, Hindi kohnī/kuhnī/kehunī, Gujarati kɔṇī and Nepali kuinu. These supply both the medial-h and contracted/metathesised comparanda for the local forms.')
for l,words in [('Malvi','ā̃kh|ānkh|ākh|āŋkh|aŋkh'),('Bagheli','ākh|ākhi')]:
 add(l,'eye',words,'43','Prakrit akkhi/acchi corresponds to Hindi ā̃kh, Bhojpuri ākhⁱ and Old Marwari ākhi in CDIAL. Cluster reduction, vowel length and optional nasalisation fit the selected forms.')
add('Nimadi','eye','ḍoḷa|ḍoḷo|doḷə|ḍoḷā','6582','CDIAL explicitly derives Prakrit ḍōla ‘eye’ and Marathi ḍoḷā ‘eye’ within the dōla ‘swinging’ family. The lexical meaning is directly attested, but Marathi contact in Nimar leaves immediate inheritance versus borrowing open.','qualified')
for l,words in [('Malvi','nāk|nākh'),('Nimadi','nāk|nākh'),('Bagheli','nak|nakh')]:
 add(l,'nose',words,'6909','Prakrit ṇakka, Hindi/Gujarati/Marathi nāk and Kumaoni nākh are explicit comparanda under *nakka. Turner’s proposed deeper *nas-ka/*nast-ka origin is uncertain; the assignment stays at the attested comparative *nakka family.')
for l,words in [('Malvi','hatheḷi|hateḷi|hatəli|hateli|hateri|hatheli|hatəḷi'),('Nimadi','hatəḷai|hateḷi|hātheḷi|hātəḷāy|hateli|hatəḷei'),('Bagheli','həṭheli')]:
 add(l,'palm',words,'14029','CDIAL gives Pali hatthatala and Hindi/Punjabi hathelī from hastatala ‘palm’. The contracted local forms fit that lexicalised compound, but Turner marks Gujarati hathelī as a Hindi loan, so the immediate inheritance/contact route remains qualified.','qualified')
for l,words in [('Malvi','muh'),('Nimadi','muh'),('Bagheli','mūh|muha')]:
 add(l,'mouth',words,'10158','Prakrit muha and Hindi muh/mũh, Bhojpuri mũh and Gujarati mõh appear under mukha ‘mouth, face’. The weakened kh and loss or retention of final h fit these selected mouth responses.')
for l,words in [('Malvi','hāth'),('Nimadi','hāt|hāth'),('Bagheli','het he')]:
 if l=='Bagheli':continue
 add(l,'arm',words,'14024','CDIAL explicitly gives Hindi/Marwari hāth ‘hand, arm, cubit’ and Marathi hāt. The arm/hand semantic range is primary-source evidence, and inherited st > tth > th/t explains the stem.')
add('Bagheli','arm','hethe|hat|hath','14024','Compare CDIAL’s Hindi hāth ‘hand, arm, cubit’, Marathi hāt and Bhojpuri hāth ‘hand, forearm’. The local e-vowel and final-e variant is retained alongside the unextended forms.','qualified')
# Numbered manifests live under canonical language directories, per skill.
for lang,proposals in props.items():
 language=inv[lang][0]['Language_ID']; dest=p.parent/language;dest.mkdir(exist_ok=True)
 path=dest/'batch-001.json';assert not path.exists(),path
 payload={'language':language,'survey':lang,'batch':1,'status':'pending-review','scope':json.loads((p/'inventory-summary.json').read_text())[lang],'proposals':proposals,'deadline':'2026-09-11T15:30:00Z','researchDirectory':str(p)}
 path.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 print(lang,len(proposals),sum(len(x['formIds']) for x in proposals))
