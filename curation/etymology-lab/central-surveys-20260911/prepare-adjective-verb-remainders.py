from research_helpers import Batch
b=Batch(19)
def a(l,g,w,p,e,t='qualified',**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier=t,**kw)
for l,w in [('Malvi','thoḍa|toḍo|thoḍo|thoḍok'),('Nimadi','thoḍo|thoḍa'),('Bagheli','thoḍəka|thoḍəke|thoḍka|ṭoḍəkə|thurka|thoṛka|thoḍika|thoḍa|thoḍi|thoṛi|thora')]:
 a(l,'few',w,'13720-2','CDIAL 13720’s -ḍ extension gives Hindi thoṛā, Marwari thoṛo and Marathi thoḍā ‘a little, few’. It is distinct from 6098 *thuḍa ‘tree trunk’. The -ka extensions and source aspiration/liquid differences remain qualified.',locator='13720, -ḍ extension')
for l,w in [('Nimadi','alag|allag|aḷag|āllāg'),('Bagheli','eleg|elge')]:
 a(l,'different',w,'700','CDIAL 700 alagna gives Pali alagga ‘not joined’, Hindi alag/algā and Old Marwari alago ‘separate’. The semantic step to different is modest; source vowel changes and contact versus inheritance remain open.')
a('Nimadi','different','nyare','404','CDIAL 404 *anyākāra gives Hindi nyārā and Gujarati nyārũ ‘different, separate’. Final -e is compatible with ordinary inflection; the reconstructed compound is explicitly Turner’s hypothesis.','straightforward')
a('Malvi','same','harika','13119','CDIAL 13119 gives Hindi sarikā/sarīkhā and Old Marwari sārikho ‘like’. The Malvi h initial is consistent with the independently documented s/h alternation, but that correspondence has not yet been checked for this word’s locality, so the proposal remains qualified.')
a('Nimadi','same','sāriko|sarikā','13119','CDIAL 13119 gives Prakrit sarikkha and Hindi sarikā/sarīkhā, Old Marwari sārikho and Gujarati sarkhũ ‘like’. The elicited same meaning is compatible with the likeness adjective.','straightforward')
for l,w in [('Malvi','śamān'),('Nimadi','samān|səmān|samāṇ'),('Bagheli','seman')]:
 a(l,'same',w,'13211','CDIAL 13211 samāna means same/alike and gives Prakrit samāṇa and Gujarati samāṇũ. The conservative form may be learned or Hindi-mediated; vowel and nasal distinctions are retained for review.')
for l,w in [('Malvi','ṭuṭo|ṭoṭo|ṭuṭio|ṭuṭā'),('Nimadi','ṭuṭel|ṭuṭlo|ṭuṭḷo|ṭuṭyo'),('Bagheli','ṭuṭeha|ṭuṭ|ṭuṭa|ṭuṭel')]:
 a(l,'broken',w,'6065','CDIAL 6065 includes the participial Prakrit tuṭṭa/ṭiuṭṭa ‘broken’, Hindi ṭūṭā and Bhojpuri ṭūṭal alongside truṭyati ‘is broken’. This selects the broken participle within the entry; regional -l/-eha endings and vowels remain for morphological review.',locator='6065, participial tuṭṭa')
for l,w in [('Malvi','phuṭo|phuṭio'),('Nimadi','phuṭa'),('Bagheli','phuṭ|phuṭe|phuṭehe|puṭ|puṭa|puṭh')]:
 a(l,'broken',w,'13845','CDIAL 13845 includes Prakrit phuṭṭa ‘burst’ and Middle/Eastern Indo-Aryan split/broken senses. These support the broken-result adjective, with final endings and deaspiration qualified. The superficially similar 13841 sphuṭa ‘clear/open’ is not selected.',locator='13845, participial phuṭṭa')
a('Malvi','it burns, it burned','baḷiyo|baḷio','6654','CDIAL 6654 *dvalati compares Old Marwari balaï, Gujarati baḷvũ and Awadhi barab ‘burn’. These selected bare stem-plus-past forms fit the intransitive family; the -iyo/-io morphology needs local review. Progressive and go-auxiliary responses are excluded.')
a('Bagheli','it burns, it burned','jeleṭh|jeleṭhe|jereṭa','5306','CDIAL 5306 gives Prakrit jalaï and Bihari/Awadhi jarab ‘burn’. The selected forms are analysed provisionally as this stem plus a -t participial/habitual ending; dental/retroflex transcription, final vowels and exact tense morphology require review. Responses with overt copulas or auxiliaries are excluded.')
b.save()
