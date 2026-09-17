from research_helpers import Batch,P
b=Batch(17)
def a(l,g,w,p,e,t='qualified',**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier=t,**kw)
for l,w in [('Malvi','akaś|akkās|ākaś'),('Nimadi','ākāś|ākāṣ|ākās'),('Bagheli','akaś|akas|akhas')]:
 a(l,'sky',w,'1008','CDIAL 1008 gives Pali/Prakrit ākāsa and contracted Prakrit āgā/āāsa. Retained k and the conservative whole form identify the ākāśa family but do not establish inherited local transmission; learned/Hindi mediation and Bagheli aspiration remain questions.')
for l,w in [('Malvi','vadro|bādal'),('Nimadi','vādəḷo|vādal')]:
 a(l,'sky',w,'11567','CDIAL 11567 compares Prakrit vaddala ‘cloud’, Gujarati vādaḷ/vādḷũ and Hindi bādal/bādar. The survey elicits sky, so this is a qualified cloud-for-sky semantic extension; contact and consonant contraction remain open.')
a('Malvi','wind','vayirā|vāyiro|vaira|vairo|bari|bairo','11497','CDIAL 11497.1 vātara specifically compares Gujarati vāyrɔ and Marathi vārā ‘wind’. The y-bearing forms fit medial-t loss; 11497.3 *vātāra is an alternative for long-vowel variants, so the subsection choice remains qualified.',locator='11497.1, compare 11497.3')
a('Bagheli','wind','behera|behhera|beyar|beyher|beyhera|beyheri','11497-3','CDIAL 11497.3 *vātāra gives Prakrit vāyāra ‘cool wind’ and Hindi bayār/bayārā. The y-bearing stem is close; Bagheli h and e-vowels require local confirmation, with 11497.1 vātara as a competing related formation.')
a('Malvi','wind','bāvo','11544','CDIAL 11544 gives Prakrit vāu, Old Awadhi bāū and Hindi bāu ‘wind’. The article cautions that vāyu and vāta are not always distinguishable in modern forms; final-vowel/v restoration in bāvo remains a local question.',locator='11544.1')
for l,w in [('Malvi','ḍori'),('Nimadi','dori'),('Bagheli','ḍori')]:
 a(l,'rope',w,'6225','CDIAL 6225 gives Prakrit dōra/dōrī/ḍōra and Hindi, Bihari, Gujarati and Marathi dorī/ḍorī ‘rope, string’. The dental/retroflex variation is already represented in the family.','straightforward')
for l,w in [('Malvi','rāsəḍi|rās|raha|rāso|raśiyo|rāśio|rasā|rasyo|rahḍi'),('Nimadi','rās')]:
 a(l,'rope',w,'10648','CDIAL 10648 raśmi compares Prakrit rassi/rāsi, Hindi rās and Gujarati rāś/rāśṛī ‘rein, string’. It explicitly traces several rassī forms through Punjabi/Hindi; local route and extended endings remain open. Malvi h-forms additionally need the s/h correspondence checked at their source localities.')
a('Malvi','rope','lej','10582','CDIAL 10582 gives Prakrit lajju, Hindi lej/lejur ‘rope’ and Bhojpuri lajurī ‘well-rope’. The l-for-r family is directly attested; no hypothetical modern Sanskrit loan is needed.','straightforward')
a('Bagheli','rope','lejuri|lejuṛi|lijuri','10582','CDIAL 10582 explicitly gives Bhojpuri lajurī ‘well-rope’ and Hindi lejur ‘rope’. The Bagheli feminine extension is compatible; vowel and r/ṛ differences remain for dialectal review.')
a('Bagheli','broom','kūcə|kuchi|kūca|kūce|kūche|kūci|kūtsi','3408','CDIAL 3408 gives Bihari kū̃cā ‘sweeper’s broom’, Nepali kuco/kuci and Gujarati kūco ‘brush’. The article entertains a Dravidian source for the older family; aspiration/ts variation and local contact remain open.')
for l,w in [('Malvi','peḍ'),('Bagheli','peḍ|pēḍ|pēḍi|pēr|pyaḍə')]:
 a(l,'tree',w,'f_rr7dv53h3a5pm','The existing *pēḍa comparative family is documented in extensions_ia.csv (Arora), with Bundeli peɖ ‘tree’ attestations in 20230522-bundeli.csv. These support a modern regional tree family, not a demonstrated Sanskrit etymon; Bagheli vowel and stop/flap variation remain qualified.',citation='arora;bundeli',source_url=str(P.parents[2]/'data/other/forms/20230522-bundeli.csv'))
a('Bagheli','tree','birbe|birba','12060','CDIAL 12060 vīrudha gives Hindi birwā ‘small plant, shrub’, Punjabi birwā ‘plant’ and Nepali biruwā ‘seedling’. The b/w correspondence and plant-to-tree generalization are plausible but remain qualified; birca is excluded.')
for l,w in [('Malvi','var|baras|vərś'),('Nimadi','varas|baras'),('Bagheli','beres|beris|bers|verś|vers')]:
 a(l,'year',w,'11392','CDIAL 11392.2 gives Prakrit varisa, Bhojpuri baris, Old Awadhi barisa, Old Marwari barasa and Gujarati varas ‘year’. Conservative varś-type forms may be learned, while contracted var and e-vocalism need local confirmation.',locator='11392.2')
b.save()
