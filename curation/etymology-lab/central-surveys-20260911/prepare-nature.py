from research_helpers import Batch
b=Batch(3);add=b.add
for l,w in [('Malvi','āg'),('Nimadi','āg'),('Bagheli','aag|ag|agi')]:
 add(l,'fire',w,'55','CDIAL explicitly gives Prakrit *aggi*, Old Marwari/Gujarati *āgi*, Hindi *āg/āgī* and Bhojpuri *āgī*. Simplification of gn through gg, with lengthening or retained final i, supports this fire family.')
for l,w in [('Malvi','dhua|dhuā̃|dhũo|dũva'),('Nimadi','dhuõ|duo|dhuā'),('Bagheli','dhuā|dhūa|dhua|ḍuā|ḍūvā')]:
 add(l,'smoke',w,'6849','CDIAL gives Hindi *dhūā̃/dhuwā̃*, Bhojpuri *dhuā̃* and Gujarati *dhūvɔ* from *dhūma*. The weakened m and vowel hiatus are supported; the Bagheli retroflex d variants and some deaspirated forms warrant local phonetic confirmation.','qualified' if l=='Bagheli' else 'straightforward',locator='6849.1')
for l,w in [('Malvi','rākh'),('Nimadi','rākh'),('Bagheli','rekh|rakh|rakhi')]:
 add(l,'ash',w,'10552','Prakrit *rakkhā*, Hindi *rākh/rākhī* and Gujarati/Marathi *rākh* are listed under the ashes homonym *rakṣā*. The meaning selects this entry, not the separate ‘protection’ homonym.')
for l,w in [('Malvi','rakhoḍi|rākhoḍi|rakhoḍo|rakhoḍa'),('Nimadi','rakhoḍi|rakhoḍo|rakhəḍo|rākoḍā|rākhoḍi')]:
 add(l,'ash',w,'10555','CDIAL reconstructs *rakṣāpuṭaka* specifically for Gujarati *rākhɔṛɔ/rākhɔṛī* ‘layers of ashes, ashes’. The Malvi/Nimadi expanded forms agree closely; possible spread from Gujarati and the exact local ḍ/ṛ outcome keep the route qualified.','qualified')
for l,w in [('Malvi','dhuḷo|duḷā|dhuro|duḷo|duḷa|dhuḷa|dhuḷ'),('Nimadi','dhuḷo|dhuḷā|dhuḷḷo|dhuḷ'),('Bagheli','dhul|dhur|dhura|dhurra|ḍul')]:
 add(l,'dust',w,'6835','The article groups *dhūli* with *dhūḍi* and explicitly cites Hindi *dhūl/dhūr*, Gujarati *dhūḷ/dhūṛ* and Marathi *dhūḷ*. These substantiate the lateral/rhotic variants; the Bagheli retroflex and deaspirated ḍul remains qualified.','qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','sona|sono|sunːo|sunːā'),('Nimadi','sonu|sono|soṇo|sonũ|sonā|sonnu|sonno|sunno'),('Bagheli','son|sona|sone|sono')]:
 add(l,'gold',w,'13519','CDIAL lists Hindi *sonā*, Gujarati *sɔnũ*, Marathi *sonẽ* and Old Marwari *sauno*, but explicitly says *suvarṇa* and *sauvarṇa* cannot usually be distinguished in NIA. This proposes the shared entry family only, retaining both ancestral alternatives.','qualified',locator='13519, 1 or 2')
 x=b.proposals[l][-1];x['parentForm']='suvárṇa / saúvarṇa';x['alternativeParentIds']=['13519-2'];x['qualification']='Main entry used as shared family reference; no choice asserted between its suvarṇa and sauvarṇa branches.'
for l,w in [('Malvi','patta|pattā|patti|patto'),('Nimadi','pətti|pātto')]:
 add(l,'leaf',w,'7733','Prakrit *patta/pattiā* and Hindi *pattā/pattī* ‘leaf’ are explicit under *pattra*. These dental-stop forms match; Malvi pān and the nasalised or retroflex-looking Nimadi forms require separate analysis.')
add('Bagheli','leaf','peṭːa|peṭːi|peṭṭa|paṭi|paṭṭi|peṭṭe','7733','Compare Hindi *pattā/pattī* and the article’s Prakrit *patta/pattiā*. The sense and structure fit *pattra*, but Bagheli’s consistently retroflex transcription must be checked as a local sound development or transcription convention; it is not silently treated as dental.','qualified')
for l,w in [('Malvi','phul|ɸul'),('Nimadi','phul|ɸul'),('Bagheli','ɸūl')]:
 add(l,'flower',w,'9092','CDIAL lists Prakrit *phulla* ‘flower’, Hindi *phūl*, Old Marwari *phūla* and Gujarati/Marathi *phūl*. The survey’s ph/ɸ variants fit this family, distinct from the fruit word *phala*.')
for l,w in [('Malvi','ɸal|phaḷ|phal'),('Nimadi','phaḷ|phal|phāḷ'),('Bagheli','phel|phəl|phəlua')]:
 add(l,'fruit',w,'9051','CDIAL gives Hindi *phal*, Gujarati/Marathi *phaḷ*, and eastern *phar*. The ph/ɸ and l/ḷ variants are direct comparanda; Bagheli’s retained l and -ua extension leave Hindi contact or local derivation open.','qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','keḷa|keḷā|keḷo|kerā|kero|kele|kelā|kela'),('Nimadi','keḷa|keḷo|keḷā|kelā'),('Bagheli','kela|keṛa|kyera')]:
 add(l,'banana',w,'2712','The first branch of *kadala/kadalī* has Prakrit *kēla*, Hindi *kelā*, Bhojpuri *kerā* and Gujarati/Marathi *keḷ-* forms. Contraction and lateral/rhotic variation support this branch; the unrelated retroflex-d form *kaḍalī* is not selected.',locator='2712.1')
for l,w in [('Malvi','kaṭ̃a|kānṭa|kāṭə|kāṭā|kā̃ṭo|kaṭ̃ā|kā̃ṭā|kānḍo|kaṇṭo'),('Nimadi','kāṭo|kāṭṭa|kāṭā|kā̃ṭo|kātːo'),('Bagheli','keṭe|kanṭa|kāṭa|kāṭē|kāṭā')]:
 add(l,'thorn',w,'2668-2','CDIAL’s *kaṇṭaka* subsection gives Hindi/Marathi *kā̃ṭā*, Marathi *kāṭā*, Gujarati *kā̃ṭɔ* and Sindhi *kaṇḍo*. These substantiate the full-vowel thorn nouns, rather than automatically using the shorter *kaṇṭa* branch; merged ‘bite’ records are excluded.',locator='2668.2')
for l,w in [('Malvi','vadəḷā|badaḷa|bādaḷā|badəḷā|badəḷ|vadəḷo|vādərā|vadəro|bādəla|baddal|vādəḷa|baddaḷ|bādəl'),('Nimadi','vādəḷa|vādəḷo|bāddal|badaḷ|vādḷā|vadəḷā|bādəḷo'),('Bagheli','beḍeṛi|beḍər|baḍer')]:
 add(l,'cloud',w,'11567','CDIAL has Prakrit *vaddala* ‘cloud’, Hindi *bādal/bādar*, Bihari *bādar* and Gujarati/Marathi *vādaḷ*. Initial v/b and lateral/rhotic variants are directly supported; Bagheli’s retroflex d notation remains a phonetic qualification.','qualified' if l=='Bagheli' else 'straightforward')
b.save()
