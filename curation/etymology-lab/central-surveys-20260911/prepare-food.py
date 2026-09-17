from research_helpers import Batch
b=Batch(4);add=b.add
for l,w in [('Malvi','gãv|gaũ|ghaũ|gavũ'),('Nimadi','gaũ|ghaũ|gahũ|gehũ|gau'),('Bagheli','gehu|gehū|gēhū|ghēh|gohū')]:
 add(l,'wheat',w,'4287','CDIAL gives Hindi *gehũ/gahũ/gohũ*, Gujarati *gahũ/ghaũ*, Marathi *gahũ* and Konkani *gaṃv* under *godhūma*. The h-retaining and more contracted western forms thus have explicit comparanda; the unrelated village homonym is excluded.')
for l,w in [('Malvi','haḷad|haldi|harad|aḷad'),('Nimadi','haḷad|hāḷdi'),('Bagheli','heḷḍi|helḍi|hereḍi|herḍi')]:
 add(l,'turmeric',w,'13992','CDIAL’s first branch gives Prakrit *haliddā/haraddā*, Hindi *haldī/harad*, Gujarati/Marathi *haḷad* and eastern *hardī*. The local lateral/rhotic variants support *haridrā*, with the h-less Malvi form and Bagheli retroflex notation retained as qualifications.','qualified' if l!='Nimadi' else 'straightforward',locator='13992.1')
for l,w in [('Malvi','lasaṇə|lesaṇi|lasaṇ|lasuṇ|lesaṇ|lasːaṇi|lahāṇ|lasːan|laśaṇ|lasuṇā|laśān'),('Nimadi','lasuṇə|lasun|lāsuṇ|lasuṇ|laśuṇ'),('Bagheli','lehesun|lehasun|lessun|lesun|lehəsun|lehsun')]:
 add(l,'garlic',w,'10990','The initial-l branch has Prakrit *lasuṇa/lasaṇa/lhasuṇa*, Hindi *lasun/lahsun/lassan*, Gujarati *lasaṇ* and Marathi *lasūṇ*. These explicitly support the contracted and h-bearing forms; learned ś-retention or contact is possible in the Malvi/Nimadi ś variants.','qualified' if l!='Bagheli' else 'straightforward',locator='10990.1')
for l,w in [('Malvi','kando|kāndā|kandā|kanda|kāndo'),('Nimadi','kānda|kāndā|kānta|kāndo')]:
 add(l,'onion',w,'2723','CDIAL specifies Hindi *kā̃dā*, Gujarati *kā̃dɔ* and Marathi *kā̃dā* ‘onion’ under *kanda* ‘bulbous root’. The source meaning therefore supports this family directly; Nimadi kānta additionally needs confirmation of the devoiced stop.','qualified' if l=='Nimadi' else 'straightforward')
for l,w in [('Malvi','tel|teḷ|tīl'),('Nimadi','tel|teḷ')]:
 add(l,'oil',w,'5958','Prakrit *tēla/tella/tilla*, Hindi *tel* and Gujarati/Marathi *tel* continue *taila* ‘oil’. Prakrit tilla also supplies a comparator for Malvi tīl without assigning the sesame-seed noun *tila*.')
add('Bagheli','oil; fat','ṭel','5958','CDIAL gives Hindi/Bihari *tel* ‘oil’ and explicitly Romani *tel* ‘oil, fat, butter’. The merged oil/fat meanings are compatible here, unlike unrelated homonyms; Bagheli’s retroflex t remains a transcription/sound qualification.','qualified')
for l,w in [('Malvi','luṇ|loṇi|loṇ|lũṇ|nuṇ|non'),('Nimadi','loṇə|loṇ'),('Bagheli','lon|loṇ|luṇ|non|noṇə|nuṇə|nun|noṇ')]:
 add(l,'salt',w,'10978','CDIAL lists Prakrit *loṇa/lūṇa*, Marwari *lūṇ*, Marathi *loṇ*, Maithili *non* and Bhojpuri *nūn*. Both the lateral and initial-n variants are explicitly represented; the distinct namak loan family is excluded.')
for l,w in [('Malvi','mas|mā̃s|mãś'),('Nimadi','mā̃s|mās|mā̃ns'),('Bagheli','mes|mash|masu|māsu')]:
 add(l,'meat',w,'9982','Prakrit *maṃsa/māsa*, Hindi/Marathi *mās/mā̃s* and Maithili *māsu* are cited beneath *māṃsa*. The nasalised and unnasalised forms have direct comparanda; merged meat/month records are excluded.')
for l,w in [('Malvi','anḍa|anḍo|inḍa|inḍo|aṇḍā|aṇḍo|anḍā|inḍu'),('Nimadi','aṇḍo|anḍo|aṇḍā|ānḍo|āṇḍo'),('Bagheli','eṇḍa|aṇḍa')]:
 add(l,'egg',w,'1111','CDIAL gives Hindi *aṇḍā*, Marathi *ā̃ḍẽ* and Gujarati *ĩḍũ* ‘egg’ under *āṇḍa*. The front-vowel western forms are explicitly supported, but Turner also identifies diffusion of Hindi aṇḍā eastward, so the Bagheli route remains qualified.','qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','gāy|gā|gāi|gayyā'),('Nimadi','gāy|gai'),('Bagheli','gey|geyye|gai|gay|gayi')]:
 add(l,'cow',w,'4147-3','CDIAL’s *gāvī* branch gives Prakrit *gāī*, Hindi *gāī*, Gujarati *gā/gāy*, Marathi *gāī* and Old Marwari *gāi*. The y/i-bearing forms specifically fit that feminine branch, rather than the masculine ox head.',locator='4147.3')
for l,w in [('Malvi','gau'),('Bagheli','geu')]:
 add(l,'cow',w,'4147-2','CDIAL’s feminine *gāvā* branch includes Punjabi *gāu* and Western Pahari *gau*. This is a provisional subsection choice: gau-like outcomes also occur under *gāvī*, and the survey lacks inflectional evidence to settle the two.','qualified',locator='4147.2, 3')
 b.proposals[l][-1]['alternativeParentIds']=['4147-3']
for l,w in [('Malvi','dud|dudh'),('Nimadi','dud|dut|dudh'),('Bagheli','ḍūḍh|ḍuḍ|ḍuḍh')]:
 add(l,'milk',w,'6391','Prakrit *duddha*, Hindi/Marwari *dūdh* and Marathi *dūdh* support *dugdha*; CDIAL also cites deaspirated *dud* and final-devoiced *dut*. Bagheli’s retroflex d notation is preserved as a qualification.','qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','siŋg|siŋ'),('Nimadi','siŋ'),('Bagheli','siŋg|siŋgh|siŋghi')]:
 add(l,'horns',w,'12583','Prakrit *siṃga*, Hindi *sī̃g*, Gujarati *sĩg* and Marathi *śī̃g* occur under *śṛṅga*. The simplified initial cluster and retained or lost final g match this family; local extended -ḍ- forms are left for a separate analysis.')
for l,w in [('Malvi','sap|sā̃p|sāp'),('Nimadi','sā̃p|sāp'),('Bagheli','sap|sāp|sāph')]:
 add(l,'snake',w,'13271','Prakrit *sappa*, Hindi *sāp/sā̃p*, Bhojpuri *sā̃p* and Gujarati/Marathi *sāp* directly support *sarpa*. The Bagheli aspirated sāph is retained provisionally pending confirmation; euphemistic animal names are not included.','qualified' if l=='Bagheli' else 'straightforward')
b.save()
