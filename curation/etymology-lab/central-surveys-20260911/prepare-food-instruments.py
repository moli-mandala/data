from research_helpers import Batch
b=Batch(22)
def a(l,g,w,p,e,t='qualified',**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier=t,**kw)
for l,w in [('Malvi','bhaṭṭā|baṭṭā|bhāṭṭā|bāṭṭa'),('Nimadi','bata|baṭṭo|baṭā|bhaṭṭo|bhaṭṭā|bhāṭṭe|bhaṭṭa'),('Bagheli','bhenṭa|bhāṭe')]:
 a(l,'eggplant',w,'9369','CDIAL 9369.1 bhaṇṭākī gives Bihari bhaṇṭā and Maithili/Awadhi bhā̃ṭā for eggplant. The source’s nasal omission, gemination and gender vowels need review; the entry places the name among potentially related regional plant-name families, not a secure remote reconstruction.',locator='9369.1')
a('Bagheli','eggplant','beygen','11503','CDIAL 11503.1 vātiṅgaṇa gives Prakrit vāiṁgaṇa, Bihari baĩgan and Hindi baigan. Bagheli e-vowels and velar sequence fit this family, with exact nasalization and local transmission open; it is distinct from the bhaṇṭā family.',locator='11503.1')
a('Malvi','millet','juar|juvār|javari|jvar|javar|juvar','10437','CDIAL 10437 yavākāra gives Prakrit juāri/jōvārī, Old Marwari juvāri and Hindi juwār/jwār. The entry explicitly identifies the sorghum/jowar grain; the survey’s broader millet gloss is preserved, not reclassified botanically without source evidence.')
a('Bagheli','millet','bejera|bejeri|bejeṛi|bejəra|bajera','9201','CDIAL 9201 *bājjara gives Hindi bājrā/bājṛā and Gujarati bājrī/bājrɔ ‘millet’. The addenda reject a proposed deeper derivation as phonologically unsupported; Bagheli e-vocalism and r/ṛ variants remain qualified.')
a('Bagheli','millet','koḍou|koḍoua','3515','CDIAL 3515 kodrava compares Hindi kodo/kodõ and Oriya kodua. The entry applies this grain name to several botanical species across languages, so the broad source millet category is retained and no precise species identity asserted.')
for l,w in [('Malvi','curi'),('Bagheli','curi|cuṛi|cura')]:
 a(l,'knife',w,'3727','CDIAL 3727’s ch-knife branch includes Prakrit churī/churiā, Hindi churī and Marathi surī. Turner stresses that these ch forms spread beyond dialect boundaries; the proposal preserves possible regional contact, source deaspiration and r/ṛ differences.',locator='3727, ch-knife forms')
for l,w in [('Malvi','mundaḍi|mundi'),('Nimadi','mundi|mūndi'),('Bagheli','muḍeri|muḍeṛi|muṇḍəṛi|muḍeriya|munḍəṛi|muneri|munḍeri|munḍəḍi')]:
 a(l,'ring',w,'10203','CDIAL 10203 mudrā gives Punjabi mundī, Sindhi muṇḍrī and Gujarati mū̃drī ‘ring’. The ultimate Iranian-loan hypothesis is already noted there; the local r/d extensions and nasal/retroflex variants need review, so the family is proposed without a single transmission path.')
for l,w in [('Malvi','ret|reti|retul'),('Nimadi','ret|retu|reti'),('Bagheli','reṭa|reṭ')]:
 a(l,'sand',w,'10816','CDIAL 10816 retra gives Hindi ret/retī, Old Marwari reta and Gujarati retī ‘sand’. The addenda emphasize limited evidence for reconstructed tr; vowel/final extensions and Bagheli retroflex ṭ remain qualified.')
for l,w in [('Malvi','bālu|baḷu'),('Nimadi','vāḷu|bālu'),('Bagheli','baluri|balu|baru|baṛu')]:
 a(l,'sand',w,'11580','CDIAL 11580 vālukā gives Prakrit vāluā, Hindi bālū/bārū and Gujarati vāḷu. Both l/r outcomes are represented; the noun branch is selected rather than 11579 vāluka ‘sandy’. Bagheli extended baluri and ṛ remain for local review.')
for l,w in [('Malvi','bijəli|vijəḷi|vijəḷā|vijəri|bijəḷi|bijaḷi|bijəḷāv'),('Nimadi','bijəḷi|bijəli|ijəḷi|bijāḷāi'),('Bagheli','bijəli|bijeri|bijili|bijkli|bijuli')]:
 a(l,'lightning',w,'11745','CDIAL 11745 vidyullatā gives Prakrit vijjullayā/vijjulī, Old Marwari bījalī, Gujarati vijḷī and Hindi bijlī/bijurī. These select the lightning compound rather than bare vidyut; initial loss, extra k and unusual endings remain qualified.')
a('Malvi','mortar','uŋkaro|ũkhrā|ukhara|ũŋkhro|oŋkili|uŋkiḷi','2360-4','CDIAL 2360.4 *udukkhala gives Prakrit ukkhala/okkhala, Bihari okhar/okharā and Gujarati ukhaḷī/ukhaṛī. These support the mortar family; nasal intrusion, medial reduction and source velar aspiration need local review. The specific ukkhala branch is used, with the older loan-origin question left open.')
b.save()
