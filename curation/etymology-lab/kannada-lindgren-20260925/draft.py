"""Research-only Lindgren Kannada proposals; never writes accepted sidecars."""
import csv,gzip,json,re,unicodedata,difflib
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==250
search=[]
for a,b in [(1,35),(36,70),(71,105),(106,140),(141,175),(176,210),(211,245),(246,250)]:search+=json.loads((P/f'discovery-{a:03}-{b:03}-results.json').read_text())
assert len(search)==250
pairs='''
1-2 d3684
3,5,7,9,11,13,15 d410
4,6,8,10,12,14,16 d1
18 d1
19 d410
20 d2041
23 d3564
26 d5488
27,32 d4205
28 d3544
31 d5528
34 d5335
35 d3491
38 d1097
39-40 d1395
42 d741
43 d5050
44 d4259
46-47 d4985
51,61 d5517
52 d2654
53 d656
56 d4616
59 d5270
60 d76
62 d282
63 d2426
64 d11
65 d1953
66 d3549
67 d4797
68 d3650
69 d1654
71 d1977
73 d3263
74 d4939
75 d784
76 d1159
77 d5430
79 d2516
80 d196
81 d156
82 d3067
85 d1571
86 d5258
87 d561
88 d5517
90 d4885
91 d2826
93 d3464
94 d4345
96-97 d1479
99 d1438
100-102 d3655
103 d4004
104 d3331
105 d4259
106-107 d2053
108 d4238
109 d502
111-112 d4715
115 d665
116 d1274
117 d1892
118 d2023
119 d3103
120 d2017
123 d410
125 d2200
126 d2115
127 d4776
128 d5151
130 d1173
132 d5160
133 d1
135 d698
137 d4677
139 d3433
140-141 d3569
143 d3559
146 d621
149-150 d1173
151 d990
153 d4694
155 d183
156 d364
157 d5474
159 d5352
163 d1774
164 d1718
165 d4275
167 d1025
168 d5024
169 d234
170 d2559
171 d3999
172 d23
173 d990
174 d5548
175 d3170
182 d4753
183 d1931
184 d215
188 d5535
190 d665
191 d1041
192 d2674
193 d4666
195 d3794
197 d910
198 d3679
199 d1839
201 d2485
203 d2495
204 d4240
205 d4085
207 d1571
208 d2410
209 d636
211 d3675
213 d2646
214 d2237
215 d1298
216 d2621
218 d530
221 d3918
222 d1
223 d3070
224 d3302
225 d3434
226 d5052
227 d805
228 d1147
230 d3633
231 d3986
232 d4711
233 d474
235 d3530
237 d3690
239 d3647
240-242 d5151
243 d8
245 d1499
246 d2591
247 d4395
248 d990
250 d3688
'''
proposals={}
for line in pairs.splitlines():
 if not line.strip():continue
 nums,parent=line.split()
 for segment in nums.split(','):
  if '-' in segment:
   a,b=map(int,segment.split('-'));ns=range(a,b+1)
  else:ns=[int(segment)]
  for n in ns:assert n not in proposals;proposals[n]=parent

def norm(s):return re.sub('[^a-z]','',unicodedata.normalize('NFD',s.lower()))
def sim(a,b):return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()
needed=set(proposals.values())|{'d4316','d3647','d4085','d2041','d4205'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in needed:articles[r['Entry_ID']]=r
assert needed<=articles.keys(),sorted(needed-articles.keys())
# Cases in which a dictionary's local form/gloss needs context beyond the short homepage hit.
special={
1:'DEDR 3684 gives Kannada nīn(u) “you” singular, with nin(n)- oblique; Lindgren nīnu is the female-referent record of the same second-person pronoun.',
2:'DEDR 3684 gives Kannada nīn(u) “you” singular, with nin(n)- oblique; Lindgren nīnu is the male-referent record of the same pronoun.',
16:'DEDR 1 lists Kannada ava “that man”; Lindgren avanu adds the masculine ending -nu to the same distal stem.',
3:'The proximal article lists Kannada ivaru “these persons”; Lindgren labels the animate plural.',
4:'The distal article lists Kannada avaru “those persons”; Lindgren labels the animate plural.',
7:'The proximal article lists ivaru as a respectful singular as well as plural; Lindgren specifies honorific female singular.',
8:'The distal article lists avaru as a respectful singular as well as plural; Lindgren specifies honorific female singular.',
13:'The proximal article includes honorific ivaru; Lindgren specifies male singular.',
14:'The distal article includes honorific avaru; Lindgren specifies male singular.',
20:'DEDR 2041 lists Kannada koñca “a little”; Lindgren konca omits the palatal mark.',
23:'DEDR 3564 gives Kannada tōḷu “arm”; Lindgren glosses it “arm or hand”, and the broader hand sense is left as a source qualification.',
27:'DEDR 4205 has the Kannada pin-/hin- “back, behind” series; Lindgren hinde “back” uses its adverbial form.',
32:'DEDR 4205 gives Kannada hind/behind forms; Lindgren hinde “behind” is the adverbial member.',
42:'DEDR 741 gives Tamil ūtu “blow by mouth” and Kannada ūdu “blow with a blowpipe”; Lindgren has the broader mouth-blowing response.',
51:'DEDR 5517 gives Kannada bēyu “be burnt, be cooked” and bēyisu “burn, cook”; Lindgren bēy is the survey stem.',
61:'DEDR 5517 includes Kannada bēyu “be cooked” and bēyisu “prepare by boiling”; Lindgren bēy “cook” is a qualified family-level match.',
63:'DEDR 2426 gives Kannada sāy “die”; Lindgren sāyi preserves the die stem with a final vowel.',
79:'DEDR 2516 lists Kannada jiḍḍu as a greasy or oily substance; Lindgren glosses it “fat (organic substance)”, a narrower sense.',
84:'',
93:'DEDR 3464 gives Kannada tēlu “float”; the Lindgren response is the same verb.',
94:'DEDR 4345 contains the Kannada hū flower/flowering stem; Lindgren hū is the simple flower form.',
101:'DEDR 3655 includes Kannada nālku/four; Lindgren nālakku has an extra vowel in the cluster.',
108:'DEDR 4238 gives Kannada hogu “enter”, with Tamil puku also “go”; Lindgren hōgu “go” is a qualified extension within this article.',
111:'DEDR 4715 gives Kannada mom-maga “grandson”; Lindgren mommaga uses the generic grandchild gloss.',
112:'DEDR 4715 gives Kannada mom-magaḷu “granddaughter”; Lindgren mommagaḷu uses the generic grandchild gloss.',
133:'DEDR 1 gives Kannada alli “there, in that place”; Lindgren glosses the locative particle simply “in”.',
139:'DEDR 3433 lists Kannada tiḷi “become clear; know”; Lindgren has the cognitive “know” sense.',
150:'DEDR 1173 gives Kannada gaṇḍu “male, man”, beside gaṇḍa “husband”; Lindgren gaṇḍu has the male-person sense.',
169:'DEDR 234 gives Kannada alla “to be not so”, including all persons and numbers; Lindgren records the negative response “not”.',
171:'DEDR 3999 gives Kannada par̤e “old, worn-out”; Lindgren haḷe is a qualified family-level match with the modern h-/lateral transcription retained.',
174:'DEDR 5548 lists Kannada bēṟe “separate, other”; Lindgren bēre loses the retroflex notation.',
182:'DEDR 4753 gives Kannada mar̤e “rain”; Lindgren maḷe uses a lateral rendering of the same rain stem.',
205:'DEDR 4085 lists Kannada pāvu “snake” and explicitly notes Tulu hāvu borrowed from Kannada; Lindgren Kannada hāvu needs this dialectal h-/p- qualification.',
227:'DEDR 805 lists Kannada ese “shoot (an arrow)”; Lindgren “throw” generalizes the projectile verb.',
239:'DEDR 3647 gives Kannada nāvu “we”, inclusive in Old Kannada and unqualified in modern Kannada; Lindgren specifically marks this record inclusive.',
240:'DEDR 5151 lists Kannada ēn “what”; Lindgren yēnu has an initial glide and final vowel.',
241:'DEDR 5151 directly lists Kannada elli “where”; Lindgren yelli has initial glide.',
242:'DEDR 5151 directly lists Kannada yār “who”; Lindgren yāru has final vowel.',
}
special={k:v for k,v in special.items() if v}
# Source-grounded holds for common loans, sense mismatches and opaque compounds.
holds={
17:'Vīḷya “betel leaf” has no secure DEDR Kannada comparison in the searched full articles.',
21:'Yella “all” has no secure exact DEDR Kannada article match; short interrogative/demonstrative stems are not enough.',
22:'Prāṇi “animal” is a prāṇa-type loan; immediate Kannada donor history is not independently established here.',
24:'DEDR 4316 brackets Kannada būdi “ash” among probable loan forms; immediate donor is not resolved.',
25:'DEDR 4316 brackets Kannada būḍi “ash” among probable loan forms; preserve the separate source spelling on hold.',
29:'Hoḍe “beat” also has “side” homonyms; no exact same-sense DEDR family was verified.',
30:'Andava “beautiful” lacks a secure same-form, same-sense DEDR family.',
33:'Hoṭṭe “belly” has no secure DEDR Kannada article match from the full text.',
36:'Hakki “bird” has a standalone dictionary form, but no verified article-level ancestry.',
37:'Mika “bird” is a separate survey variant, with no verified DEDR article.',
41:'Rakta “blood” is a Sanskrit rakta-type loan; immediate donor into this Kannada variety is unverified.',
48:'Stana “breast” is an Indo-Aryan loan; immediate donor unverified.',
49:'Usirādu “breathe” is morphologically complex, and the usir plus āḍu construction needs analysis.',
50:'Sētuve “bridge” is a setu-type loan; immediate donor unverified.',
54:'Hor “carry” has several Kannada near-homonyms; no full-article same-sense parent was verified.',
55:'The second hor “carry” record remains distinct; its parent is likewise unresolved.',
57:'Hattɯ “climb” has no secure DEDR Kannada family identified.',
58:'Daḍa “coast” has no secure same-form, same-sense DEDR family.',
70:'Ḍōlu “drum” is loan-looking and has no verified immediate donor.',
72:'Huḷa “earthworm” has no securely identified DEDR article.',
78:'Dūra “far” is a dūra-type loan; immediate donor unverified.',
83:'Pukka “feather” has no secure DEDR Kannada article for this survey form.',
84:'Halavu is glossed “few” here, but DEDR 3987 gives Kannada palavu “many, several”; the sense conflict should be reviewed.',
89:'Kaṭṭige “firewood” has no secure generic Kannada DEDR article; isolated compound matches are insufficient.',
92:'Māmsa “flesh/meat” is a māṃsa-type loan; immediate donor unverified.',
95:'Hāru “fly” has no secure Kannada DEDR article; the homepage hāru under 4091 means “look”, a homonym.',
98:'Pāda “foot” is a pāda-type loan; immediate donor unverified.',
110:'Voḷḷeya “good” does not match the cited DEDR 1017 Kannada oḷḷe closely enough without dialect context.',
113:'Hullu “grass” lacks a generic article match; DEDR 1485 has only species compounds like kāci-hullu.',
114:'Hasiru “green” has no secure Kannada DEDR article match.',
121:'Hrudaya “heart” is an hṛdaya-type loan; immediate donor unverified.',
122:'Bhāra “heavy” is a bhāra-type loan; immediate donor unverified.',
124:'Bila “hole” has same-form homonyms but no secure hole article.',
129:'Tumbā is glossed “how much” here; DEDR 3331 gives “much/full”, not an interrogative. Hold the sense mismatch.',
131:'Yajamānaru “husband” is a title/loan-based response; immediate donor and honorific construction unresolved.',
134:'Vaḷage “in” has no verified independent DEDR article; compare 135 oḷage but do not collapse source records.',
136:'Iṭṭuko “keep” is a complex keep/put form with unverified stem and auxiliary segmentation.',
138:'Moṇakālu “knee” contains kālu “leg” (DEDR 1479), but the first component remains unresolved; hold whole compound.',
142:'Yele “leaf” resembles DEDR 497 Kannada ele, but initial glide and broader dialect status need verification.',
144:'Carma “leather/hide” is a carman-type loan; immediate donor unverified.',
145:'Jaṭhara “liver” is a jaṭhara-type loan with a body-part semantic shift; immediate donor unverified.',
147:'Hēnu “louse” has no securely identified DEDR Kannada parent.',
148:'Hēṇu “louse” is a separate source spelling; no secure DEDR parent.',
152:'Bahaḷa “many” is an Indo-Aryan bahula-type loan; immediate donor unverified.',
154:'Auṣadʰi “medicine” is an auṣadhi-type loan; immediate donor unverified.',
158:'Parvata “mountain” is a parvata-type loan; immediate donor unverified.',
160:'Jāsti “much” is a likely regional loan; immediate donor unverified.',
161:'Hesaru “name” is a homonym of DEDR 3941 “green gram”; no name article was verified.',
162:'Hokkaḷu “navel” has no secure Kannada DEDR article.',
166:'Rātri “night” is a rātri-type loan; immediate donor unverified.',
176:'Mānava “person” is a mānava-type loan; immediate donor unverified.',
177:'Manuṣya “person” is a manuṣya-type loan; immediate donor unverified.',
178:'Vyakti “person” is a vyakti-type loan; immediate donor unverified.',
179:'Vobba “person” resembles one-person forms, but the source form and gender derivation need corroboration.',
180:'Handi “pig” has no secure full-article Kannada DEDR match.',
181:'Eḷe “pull” is homonymous with DEDR leaf/tenderness forms; a same-sense root was not verified.',
185:'Anna “rice” is an anna-type loan; immediate donor unverified.',
186:'Hoḷe “river” has no secure full-article Kannada etymon.',
187:'Nadi “river” is a nadī-type loan; immediate donor unverified.',
189:'Hagga “rope” has no securely linked DEDR family; the homepage direct form is unlinked.',
194:'Hēḷu “say” has no secure Kannada DEDR article identified.',
196:'Bīja “seed” is a bīja-type loan; immediate donor unverified.',
200:'Hegalu “shoulder” has no secure DEDR article identified.',
202:'Saṇṇa “small” is a separate local form with no secure DEDR article match.',
206:'Halavu “some” resembles DEDR 3987 Kannada palavu “many, several”, but the sense and onset differ; hold.',
210:'Uguḷu “spit” resembles ugi in DEDR 636, but the extra syllable and derivation need confirmation.',
212:'Nakṣatra “star” is a nakṣatra-type loan; immediate donor unverified.',
217:'Sūrya “sun” is a sūrya-type loan; immediate donor unverified.',
219:'Bāla “tail” is a vāla-type loan or a separate tail stem; immediate donor unresolved.',
220:'Togō “take” is colloquial and morphologically complex; parent stem/auxiliary unresolved.',
229:'Vēḷe “time” has a Kannada dictionary match but no secure DEDR article.',
234:'Ēḷu “wake up” is distinct from the homographic seven form in DEDR 910; the wake root is unresolved.',
236:'Voge “wash” has no secure DEDR Kannada parent.',
238:'Lindgren labels nāvu “we” exclusive, but DEDR 3647 explicitly identifies Old Kannada nāvu as inclusive and modern Kannada as unqualified; hold this exact record pending paradigm check.',
244:'Heṇḍati “wife” has no secure DEDR article match.',
249:'Nenne “yesterday” has no secure DEDR Kannada article match.',
}
assert not(set(proposals)&set(holds))

def evidence(n,r,parent):
 if n in special:return special[n]
 q=search[n-1]
 cand=[x for x in q['rows'] if x['language_id']=='Kannada' and x.get('etymon_id')==parent]
 if cand:
  cand.sort(key=lambda x:(sim(r['Form'],x['word']),int(norm(r['Gloss']) in norm(x['gloss']))),reverse=True)
  c=cand[0];word=re.sub('<[^>]+>','',c['word']).replace('|','/');gloss=re.sub('<[^>]+>','',c['gloss']).split(';')[0].strip()
  gloss=re.sub(r'^\s*\d+\.\s*','',gloss)
  if len(gloss)>100:gloss=gloss[:97]+'…'
  return f'DEDR {parent[1:]} includes Kannada {word} “{gloss}”; Lindgren {r["Form"]} “{r["Gloss"]}” matches this article-level form and sense.'
 return f'DEDR {parent[1:]} gives the corresponding Kannada family and wider Dravidian comparanda; Lindgren {r["Form"]} “{r["Gloss"]}” is assigned at article level, with its survey spelling retained.'
accepted=[];held=[]
for n,r in enumerate(records,1):
 if n in proposals:
  parent=proposals[n];ev=evidence(n,r,parent)
  accepted.append(dict(record=r,parent=parent,kind='reflex',citation=r['Source']+f';dedr[{parent[1:]}]',evidence=ev,comparison='Homepage search, full DEDR article and Lindgren source record compared; whole-article family identification.',status='pending-review'))
 else:held.append(dict(record=r,candidates=[],reason=holds.get(n,'No secure same-form, same-sense full-article DEDR match or verified immediate donor was found.')))
assert len(accepted)+len(held)==250 and len({x['record']['ID'] for x in accepted+held})==250
out=dict(status='pending-review',language='Kannada',source='lindgren2023dravidian',accepted=accepted,held=held)
for name,obj in [('proposed-decisions.json',out),('research-holds.json',held),('primary-evidence.json',articles)]:
 (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
(P/'externally-assigned-records.json').write_text('[]\n')
print(json.dumps({'records':250,'proposals':len(accepted),'holds':len(held),'parents':len(set(proposals.values())),'articles':len(articles)}))
