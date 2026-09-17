from research_helpers import Batch
b=Batch(16)
def a(l,g,w,p,e,t='qualified',**kw):
 rows=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and r['Gloss']==g]
 if rows:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rows)),p,e,tier=t,**kw)
a('Bagheli','door','ḍuar|ḍuara|ḍuari|ḍuran','6459','CDIAL 6459 *duvāra, specifically the expanded du-vowel branch, gives Prakrit duāra/duāriā, Bhojpuri duār and Awadhi duārā. Bagheli initial retroflexion and final n of ḍuran remain unexplained locally; the short-vowel *dvara head is not selected.')
for l,w in [('Malvi','saver|saverā|savere|səberā|savero|saberā'),('Nimadi','sabero|sāvero'),('Bagheli','seber|sebera|seberə|sauera|suber|subere|sūber')]:
 a(l,'morning',w,'13291','CDIAL 13291 *savēla gives Hindi sawerā ‘early morning’, Punjabi saver/saverā and Gujarati saverā. The v/b alternation is discussed in the entry; local vowel changes and broader Hindi circulation remain open.')
a('Malvi','morning','haver|havero|havera|havere','13291','The haver- forms match the saver morning family in CDIAL 13291. The existing Malvi snapshot comparison establishes s/h alternation in four independent lexical families (MALVI-S-H.md); applying it here is a qualified inference pending locality-by-locality confirmation.')
for l,w in [('Nimadi','bhansar'),('Bagheli','bhinsar|binsar')]:
 a(l,'morning',w,'11813a','CDIAL 11813a *vibhāniḥsāra explicitly gives Maithili bhinsar, Bhojpuri bhinsār and Hindi bhinsār ‘early morning, dawn’. The contracted compound has direct comparanda; Nimadi a-vocalism and Bagheli deaspiration remain local checks.')
a('Bagheli','morning','bhor','9634','CDIAL 9634 gives Hindi, Maithili, Bengali and Gujarati bhor ‘dawn’. It offers *bhōrā alongside *bhōlā, and the r form selects the former without settling the family’s deeper origin.','straightforward')
for l,w in [('Malvi','kicaḍ|kicaḍi|kicəḍ|kicːaḍ|gicaḍ|kicāḍ|kisaḍ'),('Nimadi','kicaḍ|kicāḍ|kiccaḍ'),('Bagheli','kicer|kicəḍ')]:
 a(l,'mud',w,'3153','CDIAL 3153.1 *kicca gives Gujarati kīcaṛ, Marathi kicaḍ and Kachchi kicaḍ ‘mud’. This is an expressive mud/dirt family with Dravidian comparisons, not a secure deeper Indo-European root; voiced or sibilant variants require local review.',locator='3153.1')
for l,w in [('Malvi','gara|gārā'),('Nimadi','gāro|garo|gara')]:
 a(l,'mud',w,'4137','CDIAL 4137 *gāra gives Hindi gārā ‘thick mud, mortar’ and Gujarati gārɔ ‘earth and water mixed for building’. Generic mud is compatible, but whether the source meant building mortar is unrecorded and local transmission remains open.')
for l,w in [('Malvi','jaḍ|jaḍivā|jaḍa|jaḍə|jeḍ|jaḷə|jaḍi'),('Nimadi','jhaḍə|jaḍ|jhāḍ'),('Bagheli','jeḍ|jeḍə|jeḍi|jer|jeri|jər')]:
 a(l,'root',w,'5086','CDIAL 5086.1 jaṭā already includes fibrous root; Hindi/Marwari jaṛ and Marathi jaḍ supply direct comparanda. The entry discusses Dravidian/Munda origin hypotheses. Dialect vowel, aspiration and liquid variation remain qualified; no stronger remote ancestry is claimed.',locator='5086.1')
for l,w in [('Malvi','jhāḍ|jāḍakā|jaḍəka|jāḍikā|jāḍ|jāḍiko|jhaḷi|jāḍiku'),('Nimadi','jāḍə|jhāḍə|jhāḍ|jaḍəka|jhadko|jāḍ')]:
 a(l,'tree',w,'5362','CDIAL 5362.1 jhāṭa gives Prakrit jhāḍa ‘bush, thicket’, Gujarati jhāṛ ‘tree, plant’ and Marathi jhāḍ ‘bush or tree’. The -ka forms are regional extensions; deaspiration and lateral outcomes need review. The article’s possible Munda origin is retained as uncertain.',locator='5362.1')
for l,w in [('Malvi','bāṭ|vāṭ|baṭ'),('Nimadi','vāṭ|vāṭh')]:
 a(l,'path',w,'11366','CDIAL 11366 compares Prakrit vaṭṭa, Hindi bāṭ and Gujarati/Marathi vāṭ ‘path’. It explicitly says some feminine forms could equally derive from vartis (11363), so the vartman versus vartis alternative is preserved.',locator='11366, compare 11363')
for l,w in [('Malvi','pon'),('Nimadi','pəvən'),('Bagheli','pemen|peven|pevan')]:
 a(l,'wind',w,'7978','CDIAL 7978.2 gives Prakrit pavaṇa/payaṇa and Punjabi pavaṇ/pauṇ ‘wind’. Contracted pon fits the family; the more conservative forms may be learned or regionally circulated, and Bagheli pemen’s m requires confirmation.',locator='7978.2')
for l,w in [('Malvi','barsat|varsad|bersat|varsat|vərsād|varśāt|varśat|barsad'),('Nimadi','varsāt|barsat|varsāth'),('Bagheli','bersat')]:
 a(l,'rain',w,'11398','CDIAL 11398 gives Hindi barsāt and Gujarati barsād, but explicitly allows derivation from, or contamination with, varṣartu. The rain/rainy-season metonymy is plausible; both compound analyses are retained, with source s/ś/th variation for review.')
b.save()
