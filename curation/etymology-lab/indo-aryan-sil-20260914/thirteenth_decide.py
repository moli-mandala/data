import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'6333':'CDIAL 6333 divasa explicitly lists Bshk. dōs and Maiya dis “day” and Jaunsari dūs “day/sun”, repeated in its addendum. These are direct named comparanda, including the regional day-to-sun polysemy.',
'6983':'CDIAL 6983 nava explicitly lists Bshk./Torwali nam, Palula nāwu/nā̃o and Jaunsari nō “new”. Buksa naya needs the separate navya branch rather than automatic transfer of an older nava link.',
'11567':'CDIAL 11567 vārdala gives Prakrit vaddala “cloud” and Bihari/Maithili bādar, Hindi badlī/badrā “cloud/cloudiness”. The survey badar/badri cloud forms fit this documented family; the precise intra-IA transmission remains open.',
'6328':'CDIAL 6328 dina documents din/diṇa “day, daytime” and notes learned or Hindi-mediated transmission in some languages. The survey’s sun-only meaning needs local semantic review rather than automatic reuse of a day-word assignment.',
'13415':'CDIAL 13415 sindhu explicitly lists Khowar sin and Maiya sīn “river”, alongside Shina sin and Middle Indo-Aryan sindhu. The named local comparanda support the exact river family.',
'5994':'CDIAL 5994.1 trayaḥ explicitly lists Gawri lateral-fricative λē, Kalasha trē and Khowar troi “three”; it queries the precise remodeled Khowar ending (*traye). These select branch 1, not trāyaḥ or trīṇi.',
'574':'CDIAL 574 ambā gives Middle Indo-Aryan ammā and Western Pahari ammā/āmā, with Hindi ammā “mother”. These belong to a nursery-word family; the link does not assert a uniquely recoverable origin or exclude regional transmission.',
'6835':'CDIAL 6835 dhūḍi/dhūli explicitly lists Bengali dhulā and Oriya dhuḷi “dust”. Reduplicated dhudhur forms need separate analysis of their additional material.',
'13952':'CDIAL 13952 haḍḍa gives Punjabi haḍḍī, western hāḍ and the addendum’s hāṛko/hāḍkī/hāḍgu bone forms. This supports Gojri aḍḍi with initial h-loss and Nimadi hāḍkā; it does not equate unexplained final clusters or suffixes automatically.',
'5103':'CDIAL 5103 jani explicitly gives Prakrit jaṇī and Hindi/Maithili janī “woman, wife”. The survey jani/janni forms retain that meaning and compatible consonant skeleton; regional transmission remains open.',
'10191':'CDIAL 10191 muṇḍa expressly allows muṛ spellings to represent mũṛ or the competing *muḍḍa family. Head/poll words cannot be assigned uniquely from these older links alone.',
'12548':'CDIAL 12548 śuṣka explicitly gives Palula šuko and Western Pahari śukkha/śūkho, with an -ll- extension in Torwali šugil and Oriya sukhilā “dry”. The survey simple and l-extended dry forms fit the documented family; the article retains the extension under this head.',
'4368':'CDIAL 4368 grāma explicitly gives Bshk./Gawri lām and Kalasha grom “village”. Mewari gauḍa has extra dental/retroflex material requiring separate review.',
'8399':'CDIAL 8399.1 pōta explicitly gives Bshk. pō/pɔ̄ “son, boy”. This selects the base branch, distinct from pōtara/pōtala and other enumerated extensions; the article treats the deeper origin as probably non-Indo-Aryan.',
'6663':'CDIAL 6663 dvāra explicitly gives Jaunsari dār “door” and its addendum gives Gujarati bārṇũ, supporting Nimadi bārṇu. The n-extension is included under this head; intra-IA transmission is left open.',
'3167':'CDIAL 3167 *kiyatta groups kitnā and kitrā, Old Marwari kītarāka and Western Pahari ketri/ketṇo “how much/many”. The survey katna/katra/ketra forms fit these interrogative extensions, with ordinary regional vowel remodeling.',
'5232':'CDIAL 5232 *jījja is expressly a nursery family, with Hindi jījī/jijjī/jijī “elder sister” and Marathi jijī. The family link preserves that expressive status rather than asserting a unique inherited nursery-word origin.',
'13551':'CDIAL 13551.1 sūcī explicitly gives Kalasha Urtshun sužīk “needle”, distinct from the nasal *sūñcī branch. Eastern suji needs review of the medial consonant rather than automatic reuse of an older sui assignment.',
'9917':'CDIAL 9917 maśaka lists Prakrit masa/masaa, Maithili mos and Awadhi māsā “mosquito”. Chitwan/Dang mas forms fit this documented mosquito family; transfer within Indo-Aryan remains possible.',
'3906':'CDIAL 3906.1 khura explicitly gives Kalasha and Maiya khur “foot”, alongside the hoof-to-foot extension across Dardic. The survey leg/foot responses belong to that local limb family, not the separate khuḍa branch.',
'6651':'CDIAL 6651 *dvara explicitly states that some Dardic d-initial door words may be Persian loans. Without a local discriminator, the exact inherited parent versus Persian donor remains unresolved; this exceeds merely cross-IA transmission uncertainty.',
'9882':'CDIAL 9882 markaṭa explicitly gives Khowar mukuḷ and Bshk. makīr “monkey”, questioning whether the latter continues the feminine markaṭī. That feminine is in this same entry, so the family link preserves the qualification without inventing a separate subsection.',
'3245':'CDIAL 3245 *kuḍa explicitly gives Palula kuṛī “woman, wife” and Awan kuṛī “woman”. The article debates deeper Munda/Dravidian origin; this link is to the documented Indo-Aryan family, not a claim to have settled the ultimate source.',
'1351':'CDIAL 1351 āryikā/āryakā explicitly gives Kalasha āya, Palula yēi and Torwali yäi “mother”, alongside Prakrit ajjiā “grandmother”. These named local mother-word comparanda support the parent.',
'2462':'CDIAL 2462.1 eka explicitly lists Khowar i “one”. The k-retaining Gowro form belongs to the distinct *ekka branch and is not accepted at the unsplit parent merely because an older link did so.',
'9209':'CDIAL 9209.1 *bāppa gives Punjabi bāpu “father” and the addendum Western Pahari bapu/bāpū. This selects the p-bearing nursery family, distinct from *bābba; transmission between Indo-Aryan languages remains open.'}
qs=[dict(parent=k,citation='CDIAL['+k+']',evidence=v) for k,v in E.items()];ix={q['parent']:i for i,q in enumerate(qs)}
acc=[];held=[]
for x in json.loads((P/'expanded-exact-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];reason=None
 if k in {'6328','10191','6651'}:reason=E[k]
 elif k=='6983' and r['Language_ID']=='Buksa':reason='naya requires navya-branch review, not nava by exact-link propagation.'
 elif k=='11567' and 'sky' in r['Gloss']:reason='Cloud-to-sky polysemy needs local confirmation.'
 elif k=='6835' and r['Language_ID'] not in {'B','Or'}:reason='The dust form contains additional or reduplicated material not explained by bare dhuli.'
 elif k=='13952' and r['Language_ID'] not in {'Goj','Nimadi'}:reason='Metathesis/final extension in this bone form needs local evidence.'
 elif k=='4368' and r['Language_ID']=='mewari_basad':reason='The village form has unexplained extra consonantal material.'
 elif k=='13551' and r['Language_ID']!='Kal':reason='Need to distinguish voiced medial consonant from glide, not assume identity with sui.'
 elif k=='2462' and r['Language_ID']!='Kho':reason='Retained k requires the separate ekka subsection.'
 elif re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Review source-specific uncertainty before accepting.'
 if reason:held.append(dict(record=r,families=[ix[k]],reason=reason,passNumber=13))
 else:acc.append(dict(record=r,family=ix[k],parent=k,citation=qs[ix[k]]['citation'],evidence=E[k]))
(P/'thirteenth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
(P/'thirteenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
(P/'thirteenth_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','thirteenth-decisions.json').replace('sixth','thirteenth'))
print('accepted',len(acc),'held',len(held))
