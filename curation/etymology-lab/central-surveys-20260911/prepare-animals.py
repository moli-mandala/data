from research_helpers import Batch
b=Batch(5);add=b.add
add('Bagheli','skin','cam','4701','CDIAL gives Prakrit *camma* and Hindi/Marathi *cām* ‘hide, skin’ from *carman*. This unextended form belongs to the base entry; the survey’s cemeḍ-/cemṛ- forms are separately assigned to the existing *carmaḍa-* extension node.')
for l,w in [('Malvi','macːi|machi|maci|machā'),('Nimadi','machi|macchi|macci|māchi|māci'),('Bagheli','mechi')]:
 add(l,'fish',w,'9758','CDIAL’s first branch explicitly includes Hindi *māch/māchī*, Old Marwari *maṃchā/maṃchī* and Prakrit *maccha*. These central reflexes are distinct from the *matsiya* subsection whose primary comparanda are northwestern.',locator='9758.1')
for l,w in [('Malvi','masili|mācəriyā|masiri|machəri|maciyā|macəli|mācəli|machəḷi|masḷi'),('Bagheli','meceri|mecheli|mecheṛi|mecheri|mechli|mecli')]:
 add(l,'fish',w,'9758-3','The article’s -l- extension explicitly includes Hindi *machlī*, Awadhi *macharī*, Old Marwari *māchalī* and Marathi *māsḷī*. Jambu’s dedicated *matsyal-* node represents that extension; the shortened maciyā and expanded -iyā forms retain a morphological qualification.','qualified',locator='9758.1, extension -l-')
for l,w in [('Malvi','kukəḍi|kukəḍa|kukiḍo|kukəḍo'),('Nimadi','kukəḍi|kukḍo|kukəḍa')]:
 add(l,'chicken',w,'3208','CDIAL gives Prakrit *kukkuḍa/kukkuḍī*, Hindi *kukaṛ/kukṛī* and Gujarati *kukṛɔ/kukṛī* for cock/hen. The male and female local forms continue this expressive family; the separate murg- loan responses are excluded.')
for l,w in [('Malvi','bhes|bhe|bhaisi|bhems|bhesyā|bes|bheś'),('Nimadi','bhaysi|bhāisi|bhassi|bhasi'),('Bagheli','bheyis|bheys|bheysa|bheysi|bhēys|bhesiya|bhēysi|bhēs|bhēsi')]:
 add(l,'buffalo',w,'9964','CDIAL lists Hindi *bhaĩs/bhaĩsī*, Gujarati *bhẽś* and Old Marwari *bhaïṃsa/bhaïṃsi* under *mahiṣa*. These substantiate the characteristic mh > bh and contracted vowel outcomes; Malvi bhe/bes/bhems require additional local confirmation.','qualified' if l=='Malvi' else 'straightforward')
for l,w in [('Malvi','puch|pũnc|pũc|pũch|punc'),('Nimadi','pũc'),('Bagheli','puchi|pūch|pūci')]:
 add(l,'tail',w,'8249','CDIAL gives Hindi *pū̃ch/pūchī*, Gujarati *pũch* and Prakrit *puccha/puṃcha*. Its addenda distinguish a nasalised *puñcha* variant, but Jambu currently groups it here; the proposed link retains that family-level qualification.','qualified')
for l,w in [('Malvi','puchəḍi|puchəḍo|punciḍi|putsəḍo|pucḍi|punchəḍi|pusədo|puncəḍo|pusaḍu'),('Nimadi','puchəḍā|pucəḍi')]:
 add(l,'tail',w,'8249-2','CDIAL explicitly lists a -ḍa- tail extension, with Gujarati *puchṛũ*, Hindi-region *puchaṛ* and Kumaoni *puchaṛo*. Jambu’s existing *pucchaḍa-* node represents this extension; local nasalisation, affricate weakening and suffix vowels remain qualified.','qualified',locator='8249, extension -ḍa-')
for l,w in [('Malvi','bakəri|bakari'),('Nimadi','bakari|bakəri|bākəri'),('Bagheli','bekeri')]:
 add(l,'goat',w,'9153','CDIAL cites Hindi *bakrā/bakrī*, Gujarati/Marathi *bākrā/bākrī* and Bihari *bakkar*. The reduced medial cluster and female ending fit the selected bárkara-family goat forms.')
add('Bagheli','goat','bokeri|bokeriya|bokeṛi|bokəra|bokəri','9153','CDIAL explicitly treats Hindi *bokrā/bokrī* as the bárkara family crossed with *bokka*. These rounded-vowel Bagheli forms are plausible comparanda, but the expressive/contact contribution is retained rather than calling the vowel change regular.','qualified')
for l,w in [('Malvi','katta|kuttā|kutto|kuttay'),('Nimadi','kutto|kutːo'),('Bagheli','kuṭṭe|kuṭa')]:
 add(l,'dog',w,'3275','Prakrit *kutta*, Hindi *kuttā*, Marwari *kuto* and Gujarati *kuttɔ* are explicit comparanda. Turner calls the family expressive; Malvi katta and Bagheli’s retroflex notation retain local phonetic qualifications.','qualified' if l!='Nimadi' else 'straightforward')
for l,w in [('Malvi','kuttaro|kutro|kutrā'),('Nimadi','kutro|kutrā|kutra|kutərā'),('Bagheli','kuṭera')]:
 add(l,'dog',w,'3277','CDIAL separately reconstructs *kuttira* for Gujarati *kutrɔ*, Marathi *kutrā* and Western Pahari *kutar*. The r-bearing survey forms fit that specific extension rather than the plain *kutta* head.','qualified' if l=='Bagheli' else 'straightforward')
add('Bagheli','dog','kūkura|kukra|kukur','3329','CDIAL’s first branch gives Prakrit *kukkura*, Hindi *kukur/kūkar*, Bhojpuri *kukur* and Marwari *kukro*. These k-r forms belong to kurkura/kukkura, distinct from the kutt-/kutr- families.',locator='3329.1')
for l,w in [('Malvi','bandrā|vandəro|vāndəra|bandro|bāndaro|bandar|bāndar|vandra'),('Nimadi','vāndro|vāndəro|vāndrā|vāndəriyā|bandar|bəndər'),('Bagheli','beḍer|benəra|beṇḍer|baner|baṇḍer|bāḍer|bāṇḍer|beṇḍera')]:
 add(l,'monkey',w,'11515','CDIAL gives Hindi *bā̃dar*, Marwari *bā̃dro*, Gujarati *vā̃dar/vā̃drɔ* and Bihari *bānar* from *vānara*. The b/v and presence/absence of intrusive d have explicit comparanda; Bagheli retroflexion and some contractions remain qualified.','qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','machar|macːar|machaḍa|macar|machːaḍ|macːaḷːa|masiro|maśiryā'),('Nimadi','machar|macchar|michiriyā|māchar|macəri'),('Bagheli','mecer|mecerə|mecher|mecheṛ')]:
 add(l,'mosquito',w,'9757','CDIAL gives Hindi *macchar/māchar* and Gujarati *machrũ* under the mosquito homonym *matsara*. The family match is good, but the source explicitly questions the deeper relation to *makṣā*; local r/ḍ/ḷ and extended forms require phonetic/morphological qualification.','qualified')
for l,w in [('Malvi','makhəḍi|mākaḍi|makəḍi|mākkaḍi|makaḍi'),('Nimadi','mākəḍi|mākhəḍi|mākəḍa|makəḍi'),('Bagheli','mekeḍi|mekeṛi|mekera|mekeri|makeri')]:
 add(l,'spider',w,'9883','CDIAL’s spider homonym *markaṭa* has Prakrit *makkaḍa*, Hindi *makṛī*, Bhojpuri *makarī* and Bengali *mākaṛ*. These support the medial simplification and retroflex stop/flap variants; kh-bearing forms retain an aspiration qualification.','qualified' if l!='Bagheli' else 'straightforward')
for l,w in [('Malvi','nam|nām'),('Nimadi','nāv'),('Bagheli','nēv|nav|nava|nam|ṇam')]:
 add(l,'name',w,'7067','CDIAL gives Gujarati *nām*, Old Marwari *nāva* and Marathi *nā̃v/nāv* beneath *nāman*. Both m and v outcomes are therefore supported; the emphatic particle *nāma* (7064) is a separate entry and is not the parent.')
b.save()
