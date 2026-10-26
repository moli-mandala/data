"""Review-only proposals for Konda Blair–George survey residuals."""
import csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==144
results=[]
for a,b in [(1,35),(36,70),(71,105),(106,140),(141,144)]:
 results+=json.loads((P/f'discovery-{a:03}-{b:03}-results.json').read_text())
assert len(results)==144 and all(x['count']==len(x['rows']) for x in results)
# Index: DEDR article; exact article-level comparison and any qualification.
proposals={
1:('d1634','Konda kukuṭi “hair”; survey kukti contracts the medial vowel.'),
2:('d1159','Konda kaṇ “eye”; survey kaṇaka has an expanded ending, retained as a family-level comparison.'),
3:('d1977','Konda Sova/BB gitoṇi “ear”; survey giṭoni differs only in stop notation.'),
4:('d3986','Konda pal “tooth”, alongside Kui palu and plural palka; survey palku “teeth” has the same plural stem.'),
9:('d3638','Konda nāṟu “village” with oblique nāṭ- and nāṭan “villager”; survey nāru is the village form.'),
10:('d494','Konda ilu “house”, plural ilku; survey illu has geminate l.'),
12:('d2237','Konda kōl “big wooden pestle”; survey kōlu retains the pestle word with a final vowel.'),
18:('d4523','Konda bondu “finger-ring”; survey boṇṭu has the same ring form with retroflex transcription.'),
20:('d3754','Konda nela “moon” and neleŋ “months”; survey ṇela is a moon/month record.'),
25:('d3890','Konda paṇku “stone”; survey paṇuku shows an inserted vowel.'),
26:('d2417','Konda saṟi “way, path”; survey sari matches the path sense.'),
29:('d5535','Konda vela/vēla “root”; survey vella has geminate l.'),
30:('d4345','Konda puyu, plural pūŋu “flower”; survey pūṇu corresponds to the plural flower form.'),
45:('d2146','Konda koṇes “fat” and koṛvu “fat of animals”; survey koṇuves has a related koṇ-v-s shape, a qualified family comparison.'),
46:('d4885','Konda mīn “fish”, plural mīnga; survey minku has the fish stem with plural-like velar, though the article also compares Sanskrit mīna.'),
50:('d2199','Konda kōḍi “cow”; survey kori has a weaker medial consonant transcription but the same animal sense.'),
51:('d2256','Konda kōṟi “buffalo”; survey kōvvi is a qualified family-level comparison with a different medial consonant.'),
53:('d2165','Konda goṟe “goat”; survey gure is a close goat form with a raised vowel.'),
54:('d3650','Konda nukuṛi “dog”; survey nukuḍi differs in the medial consonant notation.'),
55:('d2359','Konda saras(u) “snake”; survey sarasu matches directly.'),
56:('d2623','Konda sīma “ant”; survey sīma matches directly.'),
62:('d4616','Konda gāṛu/gālu “daughter”; survey gāṛsi shares the kin stem with an additional ending.'),
63:('d2149','Konda koṛo, with male-child plural -r; survey koṛo “boy” matches the article’s child stem and male sense.'),
67:('d5554','Konda vige “tomorrow”; survey vigehiŋ contains the tomorrow base with an added ending.'),
71:('d5320','Konda vaR- “to dry” and vaRis- causative; survey vahṭika “dry” is a qualified comparison to the dry stem.'),
73:('d2408','Konda salaŋ “cold, cool”; survey salanika contains that cold stem with an added ending.'),
76:('d1','Konda ako “distant; distance”; survey ako “far” directly matches this distal-demonstrative article.'),
78:('d410','Konda izri “small, little” occurs in the proximal article; survey ijirika shares the diminutive base with an ending.'),
80:('d1016','Konda oṛini(ka) “white, bright”; survey orinika matches the full adjective.'),
81:('d3433','Konda telani “white” and telaŋ “whitishly”; survey ṭelāŋ has the same color stem.'),
82:('d865','Konda eṟani “red”; survey eranika retains that color base with an added ending.'),
83:('d865','Konda eṟa “red”; survey eraŋ is a local red variant.'),
84:('d990','Konda unṟi “one woman or thing”; survey uṇḍre “one” retains the same numeral stem.'),
85:('d5052','Konda mūnṟi “three”; survey muṇḍri has the same numeral shape with stop transcription.'),
86:('d3655','Konda nālgi “four” for non-masculine reference; survey nallgi matches this form.'),
93:('d5151','Konda ayer/ēr “who”; survey eyēru is the plural/interrogative stem.'),
94:('d5151','Konda ayer/ēr “who”; survey eyyeru is another source spelling of the same interrogative.'),
95:('d5151','Konda ini/inika “what, what kind”; survey inika matches the non-masculine form.'),
96:('d5151','Konda ini/inika “what, what kind”; survey innika has geminate n.'),
97:('d5151','Konda em(b)e “where”; survey embe matches the article directly.'),
98:('d5151','Konda esaŋ “when”; survey esēŋ is a vowel variant.'),
99:('d5151','Konda esaŋ “when”; survey esseŋ has geminate s and a different vowel.'),
100:('d5151','Konda eso “how much”; survey essōḍ “how many” adds a terminal consonant and shifts to count.'),
101:('d5151','Konda eso “how much”; survey esso “how many” matches the interrogative quantity stem.'),
102:('d410','Konda iya/yā “this” and iyakan “this man”; survey yāka retains the proximal deictic stem.'),
103:('d1','Konda aya “that” and ayakan “that man”; survey ayāka retains the distal stem.'),
104:('d410','Konda ikeŋ “these”; survey yākēŋ has the proximal form with initial glide.'),
105:('d1','Konda akeŋ “those”; survey ayākēŋ has the distal form with initial glide.'),
107:('d5387','Konda vizu “all”; survey viju is a voiced-palatal transcription of the same quantifier.'),
108:('d5387','Konda vizer “all men, all people”; survey vijere is the corresponding personal/all form.'),
113:('d2426','Konda sā- (-t-) “die”; survey saṭān is an inflected die response.'),
114:('d2426','Konda sā- (-t-) “die”; survey saʔa is another inflected die response.'),
115:('d2426','Konda sā- (-t-) “die”; survey sāḍeŋ is an inflected die response.'),
116:('d2426','Konda sap- “kill”; survey sapṭān is an inflected kill response, not a separate lexical derivation.'),
117:('d2426','Konda sap- “kill”; survey sapʔa is another inflected kill response.'),
118:('d2426','Konda sap- “kill”; survey sapṭeŋ is an inflected kill response.'),
121:('d5514','Konda veR- “speak, tell”; survey vehṭān is an inflected speak response with rhotic/fricative notation.'),
122:('d5514','Konda veR- “speak, tell”; survey vehʔa is another inflected speak response.'),
124:('d5516','Konda ven- (veR-) “hear”; survey venʔēn is an inflected hear response.'),
125:('d5516','Konda ven- (veR-) “hear”; survey venʔa is an inflected hear response.'),
126:('d5516','Konda ven- (veR-) “hear”; survey venḍreŋ is an inflected hear response.'),
127:('d2735','Konda suṛ- “see”; survey surṭān is an inflected see response.'),
128:('d2735','Konda suṛ- “see”; survey surʔa is an inflected see response.'),
129:('d2735','Konda suṛ- “see”; survey suḍḍeŋ is an inflected see response with stop transcription.'),
130:('d5160','Konda nān(u) “I”; survey nānu matches the first-person pronoun.'),
131:('d3684','Konda nīn “you” singular; survey nīnu adds a final vowel.'),
132:('d1','Konda vānṟu “that man/he”; survey vāṇḍru is the masculine distal form with stop transcription.'),
133:('d1','Konda adi “that woman/she”; survey aḍi differs only in retroflex transcription.'),
134:('d5154','Konda māṭ(u) “we” inclusive; survey mātu is the same inclusive pronoun.'),
135:('d5154','Konda māṭ(u) “we” inclusive; survey matu is the short-vowel source spelling.'),
136:('d5154','Konda māp(u) “we” exclusive; survey māpu is the same exclusive pronoun.'),
137:('d5154','Konda māp(u) “we” exclusive; survey mapu is the short-vowel source spelling.'),
138:('d3688','Konda mīr(u) “you” plural; survey mīru directly matches.'),
139:('d1','Konda vār(u) “those men/they”; survey vāru directly matches.'),
140:('d4592','Konda bōra “chest”; survey boka differs in the medial consonant and is a qualified body-part comparison.'),
141:('d827','Konda eduṟam “chest”; survey eduram matches closely.'),
143:('d5151','Konda ēm “which” and eyek “which ones”; survey emēka is an expanded interrogative form.'),
144:('d5151','Konda ini/inika “what kind” and ayek “which ones”; survey ennika is a qualified interrogative-family comparison.'),
}
assert all(1<=n<=144 for n in proposals)
loan_holds={14:'The article’s Konda goṛel(i) “axe” is in DEDR appendix 32, which compares Indo-Aryan kuṭhāra/kuṭhārī. Survey goḍeli is close, but the immediate donor and borrowing path are unresolved.',17:'Konda pāta “cloth” in DEDR 3999 is expressly marked as from Telugu, while survey vaṭi differs in onset. Immediate donor identification remains uncertain.',21:'Ākāsam “sky” is an Indo-Aryan ākāśa-type loan; the immediate donor into this Konda dialect is unverified.',28:'Baṇāram “gold” resembles regional bangāram; no verified immediate donor attestation is identified.',41:'Lasoṇa “garlic” resembles a laśuna loan; immediate donor unverified.',43:'Beja “tomato” is a modern crop term with no secure inherited DEDR comparison or verified donor.',44:'Kobi “cabbage” is a recent crop loan with no verified immediate donor.',49:'DEDR 1680 explicitly marks Konda guḍu “egg” as borrowed from Telugu; the survey guḍḍu should wait for an exact donor-node and dialect check.',64:'Ḍinam “day” is a dina-type loan; immediate donor unverified.',65:'Rōju “day” is a regional rōju-type loan; immediate donor unverified.',68:'Vāram “week” is a vāra-type loan; immediate donor unverified.',77:'Ḍūram “far” is a dūra-type loan; immediate donor unverified.',109:'Niḍrakiḍeŋ “sleep” contains a nidrā-like stem and a verb element; donor and construction need analysis.'}
compound_holds={5:'Kilomuṇḍi “elbow” is a single opaque form; the apparent muṇḍi element does not establish a full compound analysis.',7:'Muṇṭa guṇḍakaya “heart” contains Konda guṇḍa kāya in DEDR 1693, but the initial muṇṭa and relation of the whole phrase are unresolved.',35:'Allukuṇi “potato” contains kuṇi “tuber” (DEDR 1683), but allu is unverified; hold the full compound.',36:'Baŋlaukuṇi “potato” contains kuṇi “tuber” (DEDR 1683), but the first element and compound history are unresolved.',38:'Bumi kāṭoliŋ “groundnut” is multiword; both elements and the crop-name construction need verification.',42:'Ṭelaulli “garlic” may contain the white/color stem in DEDR 3433, but ulli and the compound’s donor history remain unresolved.',57:'Mēmer koṛor “man” contains a child/man element resembling DEDR 4616/4764, but both source words require analysis before linking the whole response.',89:'Paḍakoṇḍu “eleven” is morphologically complex; its ten and one elements are not securely segmented in this dialect.',90:'Paneṇḍu “twelve” is morphologically complex and likely contains two, but the exact components and contact history need verification.',91:'Panneṇḍu “twelve” repeats the complex numeral with a second spelling; components are unverified.',106:'Uṇḍre rakam aʔeḍ “same” is a three-word response; the word rakam is loan-like and the whole predicate cannot be reduced to the one stem.'}
other_holds={22:'Ēru is glossed both “water” and “seven” in the same compiled record. DEDR 910 matches seven and 5159 matches water, but one parent would erase the homonymy.',27:'Sikla “sand” is not close enough to DEDR 575 Konda iska to assert a link; potential metathesis needs corroboration.',31:'Ṭāḍi “banana” has no verified DEDR family for this Konda form.',32:'Aṇṭi “banana” has no verified DEDR family for this Konda form.',47:'Moyeŋ “fish” resembles moya in DEDR 4737, but that article gives a particular fish and the survey gives generic fish; hold the taxonomic shift.',51:'Kōvvi “buffalo” differs substantially from DEDR 2256 Konda kōṟi; the animal match alone is insufficient.',52:'Poṭu “buffalo” could relate to DEDR 4586 pōt “male animal”, but buffalo sex and vowel are unspecified.',60:'Ṭoḍāsi “younger brother” is not sufficiently close to DEDR 3085 tambeṟsi “younger brother”; hold.',66:'Poḍoy “evening/afternoon” could be linked to DEDR 4559 podu “sun, day”, but the time-of-day development needs evidence.',69:'Koṭāka “new” has no secure Konda DEDR comparator.',72:'Usum “hot” is not close to DEDR 5517 Konda vesni “hot”; hold.',75:'Ḍagru “near” has no secure DEDR comparator.',79:'Baru “heavy” has no secure DEDR comparator.',87:'Ṭommiḍi “nine” resembles Telugu tommidi; the source dialect’s immediate donor or inherited numeral status is unresolved.',88:'Ṭomiḍi is a second nine variant resembling Telugu tommidi; origin unresolved.',110:'Basinōn “sit down” has no securely identified Konda DEDR root.',111:'Basʔa “sit down” repeats the unverified bas- stem.',112:'Basṭeŋ “sit down” repeats the unverified bas- stem.',119:'Būlaṭān “walk” has no secure direct DEDR root; broad semantic matches are insufficient.',120:'Būlʔa “walk” repeats an unverified būl- stem.',123:'Vargiḍeŋ “speak” differs from DEDR 5514 Konda veR- and 4989 miṛi; the stem and derivation need checking.',142:'Pāḍam “foot” is a pāda-type word but immediate donor and inherited status are unresolved.'}
# Keep these tempting matches on hold where the differing consonants or
# morphological analysis cannot be verified from the source article alone.
for n,reason in {
 45:'Koṇuves “fat” has the right sense but differs from DEDR 2146 Konda koṇes/koṛvu in an unexplained medial segment; the article alone does not establish its history.',
 50:'Kori “cow” differs from DEDR 2199 Konda kōḍi in the medial consonant and vowel length; a dialectal correspondence is not verified.',
 51:'Kōvvi “buffalo” differs substantially from DEDR 2256 Konda kōṟi; the animal match alone is insufficient.',
 62:'Gāṛsi “daughter” resembles DEDR 4616 Konda gāṛu/gālu, but the unexplained -si ending and distinct kin vocabulary warrant more source evidence.',
 71:'Vahṭika “dry” resembles DEDR 5320 Konda vaR-, but the hṭ versus R correspondence and adjectival ending are not verified.',
 140:'Boka “chest” differs from DEDR 4592 Konda bōra in the medial consonant; semantic agreement alone is insufficient.',
}.items():
 proposals.pop(n)
 other_holds[n]=reason
all_holds={**loan_holds,**compound_holds,**other_holds}
assert not(set(proposals)&set(all_holds))
articles={};needed={p for p,_ in proposals.values()}|{'d910','d5159','d1693','d1683','d1680','d4586','da32','d3999'}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in needed:articles[r['Entry_ID']]=r
assert needed<=articles.keys(),sorted(needed-articles.keys())
accepted=[];held=[]
for n,r in enumerate(records,1):
 if n in proposals:
  parent,comparison=proposals[n]
  evidence=f'DEDR {parent[1:]} gives {comparison}'
  accepted.append(dict(record=r,parent=parent,kind='reflex',citation=r['Source']+f';dedr[{parent[1:]}]',evidence=evidence,comparison='Homepage fuzzy search, full DEDR article and source record compared; whole-article family identification.',status='pending-review'))
 else:
  held.append(dict(record=r,candidates=[],reason=all_holds.get(n,'The homepage search and full Konda DEDR comparison yielded no secure same-form, same-sense etymon or verified immediate donor.')))
assert len(accepted)+len(held)==len(records)
assert len({x['record']['ID'] for x in accepted+held})==144
out=dict(status='pending-review',language='Konda',source='blair-george2012kondadora',accepted=accepted,held=held)
(P/'proposed-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'externally-assigned-records.json').write_text('[]\n')
print(json.dumps(dict(records=len(records),proposed=len(accepted),held=len(held),articles=len(articles))))
