import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'11392':'CDIAL 11392 varṣa section 2 lists Hindi baras and Old Marwari barasa “year”. The rain responses barkha require varṣā or a different extension; they are not automatically treated as this bare head.',
'10875':'CDIAL 10875.2 *lakkuṭa explicitly gives Hindi lakṛī “stick, firewood” and Gujarati lākṛī. Bundeli lakaḍiya is the regional feminine/plural firewood form; this corrects the older unsplit lakuṭa assignment to the k-retaining subsection.',
'3848':'CDIAL 3848 khalla lists Prakrit khallā “leather”, Bihari/Hindi khāl “hide/skin” and regional khāl. Rana khal fits this skin family, whose deeper origin and relation to challi remain uncertain.',
'5005':'CDIAL 5005 challi lists Hindi chālā “skin, hide”, borrowed into neighboring Indo-Aryan languages, and Bihari chāl/challā. The survey chala forms fit this family; their precise intra-IA route remains open under the user’s linking preference.',
'10930':'CDIAL 10930.1 *latta gives Awan lattā “clothes”, Kumauni lattā and Hindi lattā “tattered cloth”. Rana latta fits the expressive defective/rag family; this does not assert an independently settled Sanskrit origin.',
'10990':'CDIAL 10990.1 laśuna gives Prakrit lasuṇa/lasaṇa and Hindi lasun/lahsun/lassan, Marathi lasūṇ “garlic”. Initial n in Dang nasun/nahasun needs local evidence before selection over other variants.',
'10158':'CDIAL 10158 mukha explicitly gives Khowar mux “face”, with the same mouth-to-face polysemy in Middle Indo-Aryan and neighboring languages. The Khowar x spelling is directly documented.',
'7031':'CDIAL 7031.1 nasta explicitly gives Kalasha Urtshun nāst and Maiya nathūr “nose”. The latter’s extra material remains unexplained in the source, but the named local form belongs to this base section, distinct from nastī/nastu/nastya.',
'14064':'CDIAL 14064 hārdi explicitly gives Khowar hardi “heart”, analyzed through *hārdika under this same head. The survey form is a direct named comparator.',
'13878':'CDIAL 13878 syūman explicitly gives Khowar šimeni “string, rope”, while querying influence from simenu “waistband” on the vowel. The family link retains that possible contamination.',
'3023':'CDIAL 3023 kāṇḍa explicitly lists Khowar kan “tree, large bush”. Its addendum revises the deeper origin discussion toward an Indo-European comparison, so no settled Dravidian borrowing is inferred from the older main paragraph.',
'3696':'CDIAL 3696 kṣīra explicitly gives Kalasha and Khowar retroflex-affricate c̣hir “milk”. The survey affricate spellings correspond to those named local forms.',
'10713':'CDIAL 10713 *rāyaṇika “barking” explicitly gives Khowar reni “dog”. This is a direct local comparator, not a general dog-word resemblance.',
'9661':'CDIAL 9661 bhrātṛ explicitly gives Khowar brar, with its tentative development through the accusative bhrātaram, and the bhāi/bhāia family in Middle Indo-Aryan and eastern languages. Dang bhaiya is a compatible affectionate brother form; the kinship qualification is preserved.',
'12684':'CDIAL 12684 itself calls the connection of Khowar ṣoi “near” with śraya very doubtful. Uncertain cross-IA transmission does not remove this uncertainty about whether the proposed etymological family is correct.',
'2574':'CDIAL 2574 ka explicitly gives Khowar ka “who”, oblique kos. This preserves the base interrogative rather than assigning a compound kaḥ punar or kōpara parent.',
'7733':'CDIAL 7733 pattra explicitly gives Bshk. lateral-fricative paλ, Torwali pāṣ and Awan pattar “leaf”. The exact local outcomes are documented in the primary entry.',
'10506':'CDIAL 10506 *yuvatirūpa explicitly gives Jaunsari jorū “wife”, with Punjabi/Hindi jorū. This is the full historical compound etymon, already present as a persistent node.',
'6726':'CDIAL 6726 dhanus documents “bow”, including a compounded rainbow term. Bare dhanus in the rainbow slot may be an ellipsis of indradhanus; local semantic/lexical analysis is still needed.',
'13161':'CDIAL 13161 saptāha means a period of seven days and gives Nepali sātā “week”. Rajasthani sapta/sapto preserve learned pt with regional final adaptation; this family link leaves learned or neighboring Indo-Aryan transmission open.',
'9229':'CDIAL 9229 bāhu gives Hindi bā̃h “arm”, Western Pahari bā/bāi in the addendum and Bhojpuri bā̃h. Bundeli bai and Kaithal ba fit the regional contracted arm forms; intra-IA transmission is not determined.',
'1111':'CDIAL 1111 āṇḍa documents aṇḍa/ā̃ṛ egg/testicle forms and states that Hindi egg forms spread into eastern languages. The survey aḍa/ãra forms are linked provisionally to this family while their exact regional transmission remains open.',
'43':'CDIAL 43 akṣi explicitly gives Kalasha ēč and Palula ac̣hi “eye”. The survey forms match these named affricate-bearing reflexes, without relying on the article’s separately unexplained Khowar initial.',
'6298':'CDIAL 6298.1 dāru explicitly gives Khowar dar “timber, firewood”. This is the wood noun, distinct from the homophonous door family.',
'13561':'CDIAL 13561 sūtra explicitly gives Khowar šutur “thread” and discusses initial š through influence from rope or needle words. The exact local form supports the thread family with that contamination qualification.',
'125':'CDIAL 125.1 aṅgāra explicitly gives Gowro nār, Khowar aṅgār and Maiya agār “fire”. It specifically rejects interpreting the northern nār forms as Arabic-mediated Pashto loans.',
'4963':'CDIAL 4963.1 chagala explicitly gives Oriya cheḷi, Palula chēli and Bihari/Bhojpuri cherī “goat”. The survey forms fit the single-consonant branch rather than separately numbered geminate or extended stems.',
'6658':'CDIAL 6658.1 dvādaśa explicitly gives Bshk. bāh, Gawri bāš and Palula bōš “twelve”. These select the initial b- branch, not duvādaśa.',
'12707':'CDIAL 12707 *śriṣṭa explicitly gives Bshk. šiṭh/šiṭ “house”. It distinguishes Torwali śīr as more likely belonging to a different ladder/house family; no transfer of that alternative is made to Bshk.',
'12583':'CDIAL 12583 śṛṅga explicitly gives Bshk. ṣīṅ “horn”. Mewari hiŋgḍo requires independent examination of its initial change and added material.',
'6984':'CDIAL 6984 nava “nine” explicitly gives Bshk. num and Torwali nom, and Western Pahari nao/nau. These select the numeral homonym, not nava “new”.',
'9696':'CDIAL 9696 makṣā/makṣikā gives Prakrit macchī and Bengali māchi, Bihari māchī “fly”. The fly gloss distinguishes these from the similar matsya fish family.'}
qs=[dict(parent=('10875-2' if k=='10875' else k),citation='CDIAL['+('10875.2' if k=='10875' else k)+']',evidence=v) for k,v in E.items()];ix={k:i for i,k in enumerate(E)};acc=[];held=[]
for x in json.loads((P/'expanded-exact-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];reason=None
 if k in {'12684','6726'}:reason=E[k]
 elif k=='11392' and r['Gloss']!='year':reason='barkha rain requires review of a different extension or feminine varṣā parent.'
 elif k=='10990' and r['Language_ID']=='Dang':reason='Initial n instead of l/r in this garlic form needs local support.'
 elif k=='12583' and r['Language_ID']!='Bshk':reason='The horn form has initial h and a retroflex extension needing local review.'
 elif re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Review source-specific uncertainty before accepting.'
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=14))
 else:acc.append(dict(record=r,family=i,parent=qs[i]['parent'],citation=qs[i]['citation'],evidence=E[k]))
(P/'fourteenth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'fourteenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'fourteenth_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','fourteenth-decisions.json').replace('sixth','fourteenth'));print('accepted',len(acc),'held',len(held))
