"""Pending-review Muduga Varghese etymology research; no accepted graph writes."""
import csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==145
search=[]
for a,b in [(1,35),(36,70),(71,105),(106,140),(141,145)]:search+=json.loads((P/f'discovery-{a:03}-{b:03}-results.json').read_text())
assert len(search)==145 and all(q['count']==len(q['rows']) for q in search)
# Indices refer to exact source records; article nodes contain whole families.
groups=[
('d5031',[3],'DEDR 5031 gives Tamil mūñci and Malayalam muññi/moññi “face”; Muduga mūñḍźi is a local affricated face variant.'),
('d1448',[4,5],'DEDR 1448 gives Tamil and Malayalam kātu “ear”; Muduga kātɨ keeps the ear stem, recorded separately in both dialects.'),
('d3633',[6],'DEDR 3633 gives the nā-ḷ- tongue family, including Kannada nālige and southern nāḷ forms; Muduga nāḷaŋge shares the stem and extension.'),
('d2023',[7],'DEDR 2023 gives Malayalam kayyi “hand, arm” and Kannada kayyi “hand, fore-arm”; Muduga kayyɨ directly matches the arm stem.'),
('d5409',[8,9],'DEDR 5409 gives Tamil/Malayalam viral and Kannada beraḷ “finger”; Muduga biraːlɨ/bira l are source spellings of this lateral finger stem.'),
('d1904',[10,11],'DEDR 1904 gives Tamil kūrai “roof, small hut” and Malayalam kūra “hut, thatch”; Muduga kūrɛ “house” retains this hut/roof family with a broader dwelling gloss.'),
('d5440',[12,13],'DEDR 5440 gives Tamil viṟaku/virāy and Malayalam viṟuku “firewood”; Muduga birā has the same fuel sense with initial b- and reduced ending.'),
('d3045',[30,31],'DEDR 3045 gives Tamil taṇṇīr/taṇṇi and Malayalam taṇṇīr “water”; Muduga ta ṇṇi retains the water stem despite source-internal spacing.'),
('d497',[40],'DEDR 497 gives Tamil ilai and Malayalam ila “leaf”; Muduga ilɛ is a close leaf variant.'),
('d4867',[43,44],'DEDR 4867 gives Malayalam muḷaku “pepper” and Kodagu maḷu “chillies”; Muduga muḷa is the shortened chilli member of that family.'),
('d4616',[55,56],'DEDR 4616 gives Tamil makaṉ and Malayalam makan “son”; Muduga magæ is the corresponding son form.'),
('d3630',[72],'DEDR 3630 gives Tamil naṉai and Malayalam nanayuka “become wet”; Muduga nalɛndɨrkɨtɨ is an inflected wet response sharing the nasal-lateral base, with its extra verbal material retained for review.'),
('d601',[73,74,75,76],'DEDR 601 gives Tamil uṇaṅku and Malayalam uṇaṅṅuka “dry”; Muduga uṇadɨtɨ/uṇandatɨ are separate inflected dry responses of this uṇa(n)- base.'),
('d5496',[87,88],'DEDR 5496 gives Tamil veḷḷai and Malayalam veḷḷa “white”; Muduga bɛḷḷɛ keeps the color stem with b- transcription.'),
('d5151',[93,94],'DEDR 5151 lists Tamil yār/ār and Malayalam ār “who”; Muduga ārɨ is the matching interrogative.'),
('d5151',[95,96],'DEDR 5151 gives the e-/en- interrogative base and Malayalam entu “what”; Muduga ɛnda is a local what form.'),
('d5151',[97,98],'DEDR 5151 gives Malayalam eṅṅu “where”; Muduga ɛŋkɨ/ɛ̄ŋkɨ share the nasal locative interrogative stem.'),
('d5151',[99,100],'DEDR 5151 gives Tamil eppōtu and Malayalam eppōḷ “when”; Muduga ɛppɨ retains the temporal epp- base.'),
('d5151',[101,102],'DEDR 5151 gives Tamil ettaṉai and Malayalam ettira/etra “how many”; Muduga ɛttini/ɛttina share the quantity-interrogative base.'),
('d410',[105,106],'DEDR 410 gives Tamil itu and Malayalam itu “this thing”; Muduga idɨ is the proximal demonstrative.'),
('d1',[107,108],'DEDR 1 gives Tamil atu and Malayalam atu “that thing”; Muduga adɨ is the distal demonstrative.'),
('d1124',[114,115],'DEDR 1124 gives Malayalam kaṭikka “bite” and Kannada kaḍi; Muduga kaḍi is the bite stem, while the mixed present/past source gloss remains unchanged.'),
('d4091',[126],'DEDR 4091 gives Tamil pār and Malayalam pārkka “look”; Muduga parɨ is the look stem even though the elicitation also mentions a past response.'),
('d3794',[127],'DEDR 3794 gives Kannada nōḍu “look” and southern nōkku forms; Muduga nōḍɨ matches the look verb.'),
('d5160',[128,129],'DEDR 5160 gives Tamil nāṉ and Malayalam ñān “I”; Muduga nānɨ is the first-person singular pronoun.'),
('d3684',[130,131,132,133],'DEDR 3684 gives Tamil/Malayalam nī and Kannada nīn “you” singular; Muduga ni is the same 2sg stem in the source’s informal and formal elicitation slots.'),
('d1',[134,135],'DEDR 1 gives the av- distal masculine paradigm including Tamil avaṉ; Muduga avā/avɛ “he” are dialect variants with the same pronoun stem.'),
('d1',[136,137],'DEDR 1 gives the av- distal feminine paradigm including Tamil avaḷ; Muduga ava “she” is a source variant of that stem.'),
('d3647',[138,139],'DEDR 3647 gives Tamil nām with nam- oblique and Malayalam nām “we” inclusive; Muduga namma retains that inclusive nam- base.'),
('d5154',[140,141],'DEDR 5154 gives Tamil yām with em- oblique and Kannada ām/em- “we” exclusive; Muduga ɛmma directly fits the exclusive em- series.'),
('d3688',[142,143],'DEDR 3688 gives Tamil nīṅkaḷ and Kannada nīm/nīvu with nim(m)- oblique for plural “you”; Muduga nimma preserves the plural stem.'),
('d1',[144,145],'DEDR 1 gives Tamil avar and Kannada avar “they”; Muduga avarɨ matches the distal plural pronoun.'),
]
proposals={}
for parent,idxs,evidence in groups:
 for n in idxs:assert n not in proposals;proposals[n]=([parent],evidence)
# Eleven explicitly joins ten and one, unlike the simple inflectional paradigms.
for n in [91,92]:
 proposals[n]=(['d3918','d990'],'Muduga patinɔṇṇɨ “eleven” joins patiṉ- “ten in 11–19” in DEDR 3918 with onnu/one in DEDR 990; both constituent families and their order are retained.')
hold_reasons={
1:'Śarikalā “body” appears śarīra-like but the final kalā and immediate donor/compound structure are unverified.',
2:'The second śarikalā body record repeats the unresolved source form in a distinct dialect.',
14:'Muḷarɨ “mortar” resembles Malayalam ural (DEDR 651) with an additional muḷ- element; the whole form’s segmentation is unverified.',
15:'The second muḷarɨ mortar record repeats the unresolved compound-like form.',
16:'Piḍikallɨ “pestle” contains kallɨ “stone” (DEDR 1298), but piḍi- and the whole tool construction are unverified.',
17:'The second piḍikallɨ pestle record retains the same unresolved first component.',
18:'Koṭṭi “hammer” resembles a striking stem, but no exact tool-family DEDR article was verified.',
19:'The second koṭṭi “hammer” record remains on the same unresolved tool-form hold.',
20:'Koḍāli “axe” is a koḍali-type regional loan; DEDR appendix 32 compares Indo-Aryan material, but immediate donor and borrowing route are unverified.',
21:'The second koḍāli axe record retains the unresolved immediate donor.',
22:'Sūñji “needle” is a sūci-type loan; immediate donor into Muduga unverified.',
23:'The second sūñji needle record remains on the donor hold.',
24:'The third sūñji needle record has its own source identity and the same unresolved donor.',
25:'The fourth sūñji needle record has its own source identity and the same unresolved donor.',
26:'Modɨrā “ring” resembles a mudrā-type loan; immediate donor unverified.',
27:'The second modɨrā ring record retains the unresolved immediate donor.',
28:'Nerā “sun” has no securely matching southern DEDR article; near-sounding day/light words do not establish ancestry.',
29:'The second nerā sun record remains unresolved.',
32:'Bæni “river” has no secure same-sense DEDR article for its exact stem.',
33:'Bani is the second dialect river variant with no verified DEDR article.',
34:'Bīmamullɨ “rainbow” seems to contain Bhīma plus a bow element; immediate donor and compound analysis unverified.',
35:'The second bīmamullɨ rainbow record remains on the same compound hold.',
36:'Kukkɛ “ash” has no secure DEDR article with both matching form and meaning.',
37:'The second kukkɛ ash record remains unresolved.',
38:'Sorṇam “gold” is a svarṇa-type loan; immediate donor unverified.',
39:'The second sorṇam gold record retains the unresolved donor.',
41:'Godimbɛ “wheat” is a regional godhuma-type loan; immediate donor unverified.',
42:'The second godimbɛ wheat record retains the unresolved donor.',
45:'Bɛḷḷɛbɛŋkaya “garlic” is compound-like, with a white element and onion name; its second constituent and borrowing history are unverified.',
46:'The second bɛḷḷɛbɛŋkaya garlic record repeats the unresolved compound.',
47:'Beŋkɛya “onion” has no securely verified inherited DEDR parent or immediate donor.',
48:'Beŋgiya “onion” is a second dialect form with unresolved origin.',
49:'Ta kkāḷi “tomato” is a regional crop loan; immediate donor unverified.',
50:'The second ta kkāḷi tomato record retains the donor hold and source-internal spacing.',
51:'Moṭakos “cabbage” is a recent crop loan; immediate donor unverified.',
52:'Muṭṭakōsɨ “cabbage” is a second recent-loan form; immediate donor unverified.',
53:'Arima “mosquito” has no secure full DEDR article matching its form and sense.',
54:'Arimɛ is a second mosquito dialect form without a secure parent.',
57:'Āṇayitɛ “boy” may contain āṇ “male”, but the remaining material and whole-word derivation are unresolved.',
58:'Jama “night” is loan-like; immediate donor and sense history unverified.',
59:'The second jama night record retains the unresolved donor.',
60:'Madɛna “noon” resembles a madhyāhna-type loan; immediate donor unverified.',
61:'The second madɛna noon record retains the unresolved donor.',
62:'Vāra “week” is a vāra-type loan; immediate donor unverified.',
63:'Bārā “week” is a second dialect loan form; immediate donor unverified.',
64:'Māsa “month” is a māsa-type loan; immediate donor unverified.',
65:'The second māsa month record retains the unresolved donor.',
66:'Varsa “year” is a varṣa-type loan; immediate donor unverified.',
67:'The second varsa year record retains the unresolved donor.',
68:'Barsa “year” is the second dialect variant of a varṣa-type loan; donor unverified.',
69:'The second barsa year record retains the unresolved donor.',
70:'Mosa “bad” is loan-like; immediate donor and exact meaning of the borrowed stem unverified.',
71:'Mośa is the second dialect bad variant, with unresolved donor.',
77:'Ujarā “long” resembles the high/tall family DEDR 646, but length versus height and its ending need verification.',
78:'Immaṭi “short” appears measure-based; stem and construction are not verified.',
79:'Oru maṭi “short” is an explicit two-word measure expression; do not label it as a single inherited stem.',
80:'Balasāri “right” combines a right-side base (DEDR 5276) with an unverified sāri element; hold full expression.',
81:'Iḍasāri “left” combines a left base (DEDR 449) with unverified sāri; hold full expression.',
82:'Dūrā “far” is a dūra-type loan; immediate donor unverified.',
83:'The second dūrā far record retains the unresolved donor.',
84:'Kanadasti “heavy” has a possible kana weight stem and additional material; segmentation unverified.',
85:'Rombakana “heavy” is a compound-like quantity response; components and stem history unverified.',
86:'Kanakammi “light” may be a lesser-weight construction; segmentation and source sense unverified.',
89:'Corɛ “red” resembles the earlier Muduga śorɛ “blood” under DEDR 2883, but a blood-to-red derivation is not independently established.',
90:'Tsorɛ is the second red variant with the same unresolved blood-color derivation.',
103:'Ɛcala “what kind?” contains an e- interrogative base but the second element is not verified; hold the whole form.',
104:'The second ɛcala record repeats the unresolved second element.',
109:'Ore ːmātiri “same” combines a one/same stem with mātiri “like/kind”; donor and full compound history unresolved.',
110:'Orɛ oppa “same” is a two-part response whose oppa element is not securely linked.',
111:'Orittini “few” may contain an or- one stem, but its ending and quantity development need analysis.',
112:'Motta nā “many” is a multiword quantity response with unverified components.',
113:'The second motta nā record repeats the unresolved multiword response.',
116:'Pajipiḍikɨtɨ “hungry” contains a paci-like hunger stem (DEDR 3825) plus additional verbal material; hold whole predicate.',
117:'The second pajipiḍikɨtɨ hunger record preserves the unresolved complex predicate.',
118:'Vɛhɨkɨtɨ “thirsty” has an unidentified predicate base and tense material.',
119:'The second vɛhɨkɨtɨ thirsty response remains unresolved.',
120:'Kattɨtɨ “burn” has no secure same-sense DEDR base; kaṭṭu “tie” is a misleading homonym.',
121:'The second kattɨtɨ burn response has the same unresolved base.',
122:'Śākabeḍā is glossed both “don’t die” and “he died”; although sā- “die” is known, the negative/tense construction is unresolved.',
123:'The second śākabeḍā record repeats the mixed and contradictory elicitation response.',
124:'Kollatɛ “don’t kill, he killed” contains a kol- kill stem, but negative and past interpretation of the complete form are unresolved.',
125:'Kollabeḍā “don’t kill, he killed” contains kol- plus a prohibition marker; the conflicting source gloss needs review before a complete assignment.',
}
needed={p for ps,_ in proposals.values() for p in ps}|{'d651','d1298','da32','d5276','d449','d2883','d3825','d2132'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in needed:articles[r['Entry_ID']]=r
assert needed<=articles.keys(),sorted(needed-articles.keys())
accepted=[];held=[]
for n,r in enumerate(records,1):
 if n in proposals:
  parents,evidence=proposals[n]
  item=dict(record=r,parent=parents[0],kind='component' if len(parents)>1 else 'reflex',citation=r['Source']+';'+';'.join(f'dedr[{p[1:]}]' for p in parents),evidence=evidence,comparison='Homepage exact search, full DEDR article, source item and dialect record checked; whole-article family identification.',status='pending-review')
  if len(parents)>1:item['components']=parents
  accepted.append(item)
 else:
  assert n in hold_reasons,n
  held.append(dict(record=r,candidates=[],reason=hold_reasons[n]))
assert len(accepted)+len(held)==145 and len({x['record']['ID'] for x in accepted+held})==145
out=dict(status='pending-review',language='Muduga',source='varghese2015palakkad',accepted=accepted,held=held)
for name,obj in [('proposed-decisions.json',out),('research-holds.json',held),('primary-evidence.json',articles)]:
 (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
(P/'externally-assigned-records.json').write_text('[]\n')
print(json.dumps(dict(records=145,proposed=len(accepted),rows=sum(len(x.get('components',[x['parent']])) for x in accepted),held=len(held),articles=len(articles))))
