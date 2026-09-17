import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'9828':'CDIAL 9828 explicitly gives Bshk. mīš “young man, husband” with plural mānuš under manuṣya, while discussing contamination with puruṣa/purisa. Khowar moš also has competing mānuṣa/martya analyses, so those records are held.',
'10924':'CDIAL 10924 leaves many larikā/laṛkā forms between *laḍikka and *laḍḍikka with shortening. An older identical link does not settle that branch choice.',
'6423':'CDIAL 6423 dur/duraḥ explicitly names Kalasha dūr and Khowar dur “house”. The historical door-to-house sense is therefore directly documented.',
'9361-2':'CDIAL 9361.2 bhagna explicitly derives run/flee stems through Prakrit bhagga “broken, fled”, with Hindi bhāgnā and Old Marwari bhāgaï “runs”. The article retains the debated relation between breaking and fleeing; this selects the g-stem branch.',
'3197':'CDIAL 3197 kīdṛśa gives Marathi kasā, Old Marwari kiso and ka-replacement forms in Apabhramsha kaïsa/Hindi kaisā “of what kind?”. The survey forms have the same interrogative adjective meaning; vowel remodeling is retained rather than asserted to be mechanically regular.',
'612':'CDIAL 612 itself questions Khowar yor from aru and explicitly asks where initial y comes from. The old link is not sufficient evidence to settle that analysis.',
'7059':'CDIAL 7059 *nānna is a kinship/nursery family compared with Vedic nanā “mother” and Prasun nan “mother”. Khowar nan has the same form and meaning; this provisional family link does not establish a unique nursery-word origin or exclude regional transmission.',
'3084':'CDIAL 3084 kāla explicitly lists Bshk. and Torwali kāl “year”. It discusses Sanskrit influence on the temporal family; this link leaves that transmission history open.',
'5754':'CDIAL 5754 tāta explicitly lists Khowar tat “father”, alongside related affectionate kinship terms. The direct named comparator supports this family.',
'6481':'CDIAL 6481 duhitr explicitly gives Gawri zū and Khowar žūr “daughter”, discussing irregular palatalization or influence from jāta. The comparative family is retained with that historical qualification.',
'6236':'CDIAL 6236 explicitly questions whether Bshk. dā and related thread words are borrowed from Persian dasa rather than inherited from daśā. This is an Indo-Iranian parent/donor ambiguity, not merely cross-IA transmission.',
'10702':'CDIAL 10702 rātrī explicitly gives Maiya rāl “night”; the unusual final lateral is directly documented rather than inferred from a plains form.',
'12497':'CDIAL 12497 śīrṣa explicitly gives Kalasha ṣiṣ and Maiya šiš “head”, selecting this head/skull branch rather than śiras.',
'6331':'CDIAL 6331 diva explicitly gives Kalasha dī “sky” and Torwali dī “day”; both survey meanings have named primary comparanda.',
'11348':'CDIAL 11348 *varta explicitly gives Bshk. baṭ “stone”, while Khowar is bort. Nonmatching Khowar boht and Mewari bhāṭo require further local phonological/branch review.',
'6590':'CDIAL 6590 doṣā explicitly gives Kalasha and Khowar doṣ “yesterday”; the temporal sense develops from night/evening in the named comparanda.',
'7641':'CDIAL 7641 *pakṣya explicitly gives Khowar peṭṣ/pec̣ “hot” and Kalasha pēc̣ī “heat”. The survey retroflex affricate spelling is retained.',
'9187':'CDIAL 9187 bahu explicitly gives Khowar bo(h) “many”, confirmed again in its addendum. This is a direct local comparison.',
'12452':'CDIAL 12452 śiras gives Lahnda/Punjabi sir and Gujarati sir/sar “head”. Awan ser is a compatible regional vowel variant in the same head family; local transmission remains open.',
'8179':'CDIAL 8179 pitṛ explicitly gives Awan pio and Lahnda peo “father”. The same pio in Pothwari belongs to this documented regional family.',
'12326-2':'CDIAL 12326.2 *śarṇa explicitly gives Bshk. šan “roof”, distinguishing it from the uncontracted śaraṇa branch.',
'3219':'CDIAL 3219 *kuccura explicitly gives Bshk. kučur and Chilis kučuro, with Maiya kūsar and Kanyawali kučara. The survey forms preserve the affricate-bearing dog family; no connection to the separate kurkura family is asserted.'}
qs=[dict(parent=k,citation='CDIAL['+k.replace('-','.',1)+']',evidence=v) for k,v in E.items()];ix={q['parent']:i for i,q in enumerate(qs)};acc=[];held=[]
for x in json.loads((P/'remaining-exact-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];i=ix[k];reason=None
 if k in {'10924','612','6236'}:reason=E[k]
 elif k=='9828' and r['Language_ID']!='Bshk':reason='The man-word has competing manuṣya/mānuṣa/martya or local morphology analyses; exact older links do not settle the parent.'
 elif k=='11348' and r['Language_ID']!='Bshk':reason='The stone form differs from the article’s local comparator; resolve local phonology and branch before saving.'
 elif re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source uncertainty requires distinguishing lexical reading from locality attribution.'
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=12));continue
 acc.append(dict(record=r,family=i,parent=k,citation=qs[i]['citation'],evidence=E[k]))
(P/'twelfth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
(P/'twelfth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','twelfth-decisions.json').replace('sixth','twelfth');(P/'twelfth_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
