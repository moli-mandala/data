"""Explicit native first reading; immutable source evidence remains attached.

No installation. Page coverage expands only after reading its complete native text;
all forms still need visual verification against the original.
"""
import json
import re
import unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
SURFACES={
149:[(22,33),(35,43),(46,56),(58,66)],
150:[(20,26),(28,36),(39,50)],
151:[(5,19),(24,34),(36,41),(44,[57,58]),(63,73),(75,86)],
152:[(4,12),(27,32),(35,44),(46,[54,55])],
153:[(5,17),(20,32)],
154:[(11,20),(23,33),(39,49),(52,59),(62,74)],
155:[(19,[40,41]),(43,[62,63])],
156:[(4,27),(30,39),(49,69),(71,[85,86])],
157:[(5,24),(36,50),(53,67),(69,84)],
158:[(46,57)],
159:[(5,16),(18,25),(30,40),(42,50),(56,63)],
160:[(5,14),(16,28),(33,40),(51,62),(64,71)],
161:[(12,19),(21,33),(36,43),(45,53),(60,70)],
162:[(4,12),(14,24),(26,34),(46,56),(58,66)],
163:[(4,11),(27,[47,48]),(50,66)],
164:[(4,18),(20,31),(40,61),(63,83)],
166:[(7,18),(20,36),(38,47),(50,60),(68,82)],
167:[(4,12),(14,22),(29,37),(39,[50,51]),(53,61)],
168:[(8,18),(21,31),(33,[49,50]),(62,70)],
169:[(10,[36,37]),(46,56),(59,68)],
170:[(9,19),(21,31),(33,40)],
171:[(5,12),(14,23),(25,31),(33,44),(46,57),(60,73)],
172:[(9,26),(28,[45,46])],
173:[(53,60),(62,72),(74,82)],
174:[(4,9),(11,14),(26,32),(35,40),(42,50)],
175:[(15,23),(25,35),(37,51)],
176:[(5,16),(18,[31,32]),(34,44),(51,58),(60,65),(67,73)],
177:[(4,[21,22]),(33,41),(43,51),(53,68)],
179:[(11,20),(22,30),(33,43),(45,58),(60,69)],
180:[(4,15),(18,25),(27,37),(53,58)],
181:[(4,11),(13,20),(28,39),(41,51),(54,72),(74,92)],
182:[(12,23),(25,35),(37,47),(49,61)],
183:[(5,20),(22,[36,37]),([39,40],[69,70,71])],
184:[(7,12),(15,23),(26,37)],
185:[(7,17),(20,30),(32,42)],
186:[(139,148),(150,166),(168,177)],
188:[(9,14),(28,35),(37,45)],
189:[(5,12),(14,21),(35,48),(50,57)],
190:[(4,14),(17,26),(28,40),(54,[70,71])],
191:[(23,32),(34,43),(46,58)],
192:[(4,12),(49,53),(55,62)],
193:[(5,10),(12,18),(28,37),(39,56),(59,70)],
194:[(4,12),(18,25),(27,34),(41,49),(51,65)],
195:[(4,14),(24,34),(36,49),(51,[67,68])],
196:[(5,16),(19,28),(30,42)],
197:[(5,23),(25,34),(39,49),(51,59),(64,77),(79,90)],
198:[(5,[21,22]),(36,[48,49]),(51,58)],
199:[(5,14),(16,25),(27,43),(45,54),(61,70),(72,82)],
200:[(11,21),(23,29),(34,[44,45])],
201:[(5,[21,22]),(27,35),(43,54),(56,65),(67,74)],
202:[(8,18),(20,[27,28]),(32,39),(41,52),(58,66)],
203:[(6,17),(24,37),(39,48),(50,61)],
204:[(4,17),(19,29),(32,48),(50,63)],
}
TABLES = {
173:[(12,'1sg','Chitwan'),(13,'1pl','Chitwan'),(16,'1sg','Tanahun'),(17,'1pl','Tanahun'),
     (20,'2sg non-honorific','Chitwan'),(21,'2pl non-honorific','Chitwan'),
     (24,'2sg non-honorific','Tanahun'),(25,'2pl non-honorific','Tanahun'),
     (28,'2sg honorific','Chitwan'),(29,'2pl honorific','Chitwan'),
     (32,'2sg high-honorific','Chitwan'),(33,'2pl high-honorific','Chitwan'),
     (36,'3sg non-honorific','Chitwan'),(37,'3pl non-honorific','Chitwan'),
     (40,'3sg non-honorific','Tanahun'),(41,'3pl non-honorific','Tanahun'),
     (44,'3sg honorific','Chitwan'),(45,'3pl honorific','Chitwan')],
178:[(20,'1sg Set1','Chitwan'),(21,'1sg Set2','Chitwan'),(23,'1pl Set1','Chitwan'),(24,'1pl Set2','Chitwan'),
     (26,'2sg Set1','Chitwan'),(27,'2sg Set2','Chitwan'),(29,'2sg honorific Set1','Chitwan'),(30,'2sg honorific Set2','Chitwan'),
     (32,'2pl Set1','Chitwan'),(33,'2pl Set2','Chitwan'),(35,'3sg Set1','Chitwan'),(36,'3sg Set2','Chitwan'),
     (38,'3sg honorific Set1','Chitwan'),(39,'3sg honorific Set2','Chitwan'),(41,'3pl Set1','Chitwan'),(42,'3pl Set2','Chitwan')],
}
# Printed examples with directly adjacent quoted glosses, including analytically
# segmented forms. Both source layers are kept instead of normalizing one to the other.
PAIRS={
146:[(30,'ɡʰuɡʰa-lə','throw-PST'),(34,'ɡʰuɡʰa-tihin','throw-PROG'),(37,'ɡʰuɡʰa-te','throw-SIM')],
147:[(5,'de-','give'),(6,'de-lə','give-PST'),(9,'de-tihin','give-PROG'),(14,'de-te','give-SIM'),
     (17,'kəhə-','say'),(23,'kəhə-lə','say-PST'),(29,'kəhə-tihin','say-PROG'),(35,'kəhə-te','while saying'),
     (49,'a-','come'),(51,'əilə','came'),(54,'a-lə','come-PST'),(57,'əitihin','coming'),
     (60,'a-tihin','come-PROG'),(63,'əite','while coming'),(65,'a-te','come-SIM')],
148:[(4,'cʰi-','touch'),(10,'cʰiulə','touched'),(14,'cʰi-lə','touch-PST'),(18,'cʰiutihin','touching'),
     (21,'cʰi-tihin','touch-PROG'),(23,'cʰiute','while touching'),(25,'cʰi-te','while touching'),
     (33,'pəlṭ-','fall'),(35,'palṭa-','cause to fall'),(36,'ləḍ-','fall'),(38,'laḍa-','cause to fall'),
     (39,'dzʰər-','fall down'),(40,'dzʰara-','fall down'),(49,'dʰus-','beat'),
     (53,'dʰus-‘lə','beat'),(58,'dʰus-tihin','beat-PROG'),(63,'dʰus-te','beat-SIM'),
     (73,'ɡokʰr-','bathe'),(77,'ɡokʰrələ','bathed')],
149:[(4,'ɡokʰr-lə','bathe-PST'),(6,'ɡokʰritihin','bathing'),(8,'ɡokʰr-tihin','bathe-PROG'),
     (11,'ɡokʰrəte','while giving'),(13,'ɡokʰr-te','bathe-SIM')],
192:[(29,'deike','to give'),(33,'səkike','to give'),(39,'rəhike','to remain'),(42,'dzaike','to give')],
196:[(53,'səkike','to finish'),(56,'deike','to give'),(59,'laɡike','to apply'),
     (61,'pərike','to fall'),(64,'aṭəi','AUX.NPST'),(67,'bʰəike','to become')],
}
# Prose target attestations and quoted morphemes; source qualifications remain
# explicit rather than inferred from shape or English free translation.
PROSE=[
(146,[26],'ɡʰuɡʰa-','throw','verb stem','Verb type 1a.'),
(146,[26],'paka-','cook','verb stem','Verb type 1a.'),
(146,[26],'laha-','bathe','verb stem','Verb type 1a.'),
(146,[26],'nida-','sleep','verb stem','Verb type 1a.'),
(146,[28],'ɡʰuɡʰa-','throw','verb stem',''),
(146,[41],'uḍʰe-','fly','verb stem','Vowel-final stem class.'),
(146,[41],'de-','give','verb stem','Vowel-final stem class.'),
(146,[41],'pure-','make something reach','verb stem','Vowel-final stem class.'),
(147,[37],'kəhə-','say','verb stem',''),
(147,[38],'bohə','float','verb stem','Printed bohə/buhə- grouping; no automatic variation edge.'),
(147,[38],'buhə-','float','verb stem','Printed bohə/buhə- grouping; no supplied hyphen on the first alternate.'),
(147,[38],'cuhə','drop/leak','verb stem',''),
(147,[38],'rəhə-','remain','verb stem',''),
(147,[40],'ɡa-','sing','verb stem','Author includes this stem in the preceding class despite its different shape.'),
(147,[44],'a-','come','verb stem','Verb stem type 2.'),
(147,[44],'dza-','go','verb stem','Verb stem type 2.'),
(147,[44],'kʰa-','eat','verb stem','Verb stem type 2.'),
(147,[46],'-i','','epenthetic','Author explicitly describes an epenthetic vowel, not a grammatical morpheme.'),
(147,[68],'cʰi-','touch','verb stem','Verb stem type 3.'),
(147,[68],'pi-','drink','verb stem','Verb stem type 3.'),
(147,[69],'dʰo-','wash','verb stem','Verb stem type 3.'),
(147,[69],'ro-','cry','verb stem','Verb stem type 3.'),
(147,[69],'pa-','get','verb stem','Author includes this stem in the preceding inflectional class.'),
(148,[29],'pəlṭ-','fall','verb stem','Verb stem type 4; source describes vowel harmony in causative formation.'),
(148,[29],'ləḍ-','fall','verb stem','Verb stem type 4; source describes vowel harmony in causative formation.'),
(148,[29,30],'dzʰər-','fall down','verb stem','Verb stem type 4; source describes vowel harmony in causative formation.'),
(148,[43],'pucʰ-','ask','verb stem','Consonant-final stem class.'),
(148,[43],'dʰus-','beat','verb stem','Consonant-final stem class.'),
(148,[44],'maɡ-','ask','verb stem','Consonant-final stem class.'),
(148,[44],'kʰor-','bathe','verb stem','Consonant-final stem class.'),
(148,[44],'pʰor-','break','verb stem','Consonant-final stem class.'),
(148,[45],'laha-','bathe','verb stem','Source lists this vowel-final form in the consonant-final section; preserve that source inconsistency.'),
(148,[67],'əlp-','disappear','verb stem','Verb stem type 5.'),
(149,[14],'rəhə-','remain/COP.PST','verb stem',''),
(149,[15],'rəhə-','remain','verb stem','Source distinguishes main verb and modal auxiliary use from copular inflection.'),
(149,[17],'həkʰə-','COP/AUX.NPST','verb stem copula auxiliary',''),
(149,[17,18],'aṭə-','COP/AUX.NPST','verb stem copula auxiliary','U+F041 has a previously reviewed ordinary-a glyph mapping; original check still required here.'),
(149,[18,19],'rəhə','','verb','Source explicitly repeats this stem in the discussion of main, compound and auxiliary uses.'),
(149,[67,68],'rəhə','','verb','Source contrasts tense-marked main verb use and suppletive auxiliary use.'),
(150,[17],'-hal','','suffix pret','Source states this past marker is followed by agreement features.'),
(150,[17],'-lə','','suffix pret','Source states this past marker occurs otherwise.'),
(152,[15],'-tə','','suffix non-past','Source subsequently specifies first and second person use.'),
(152,[15],'-tahə','','suffix non-past','Source subsequently specifies second and third person plural use.'),
(152,[15],'-ta','','suffix non-past','Source subsequently specifies first plural or object-encoding use.'),
(152,[15],'-t','','suffix non-past','Source subsequently specifies third person singular use.'),
(152,[16],'-tə','','suffix non-past','Explicit first and second person context.'),
(152,[17],'-ta','','suffix non-past','Explicit first plural or object-encoding context.'),
(152,[18],'-tahə','','suffix non-past','Explicit second and third person plural context.'),
(152,[19],'-t','','suffix non-past','Explicit third person singular context.'),
(152,[20],'-ta','','suffix non-past','Transitive verb registering two referents.'),
(152,[20],'-tai','','suffix non-past','Transitive verb registering two referents.'),
(154,[5],'-te','','suffix','Explicit habitual aspect marker.'),
(154,[6],'rəhə-','AUX.PST','stem auxiliary pret','Habitual construction.'),
(154,[6],'kər','do','verb','Habitual construction.'),
(154,[7],'-te','','suffix','Past habitual construction with the auxiliary.'),
(154,[7,8],'rəhə-','AUX.PST','stem auxiliary pret','Past habitual construction.'),
(154,[34],'-te','','suffix','Present habitual construction.'),
(154,[35],'kərike','to do','verb inf','Used in the present habitual construction.'),
(154,[35],'-larə','','suffix prospective','Compounded with kər to mark non-past habitual.'),
(154,[36],'kər','do','verb','Used in the non-past habitual construction.'),
(154,[77,78],'-t','','suffix','Source describes a non-perfective element across progressive, habitual and simultaneous-converb suffixes; not a general tense inference.'),
(155,[9],'-tin','','suffix progressive','Source gives -tin/tihin and contrasts discourse use with careful speech.'),
(155,[9],'tihin','','affix progressive','Second alternative lacks a printed initial hyphen.'),
(155,[10],'rəhə-','','stem copula pret','Source explicitly identifies past copular use.'),
(155,[10],'baṭə-','','stem copula non-past','Source explicitly identifies non-past copular use.'),
(155,[10],'aṭə-','','stem copula non-past','Source explicitly identifies non-past copular use.'),
(155,[11],'-tihin','','suffix progressive','Source specifies very careful speech.'),
(155,[12],'-tin','','suffix progressive','Source says this also occurs in discourse data.'),
(156,[41],'rəhə-','remain','stem verb','Source describes continuation by compounding with this vector verb.'),
(156,[44],'-əu','','suffix','Source identifies imminent aspect and requires verb reduplication.'),
(157,[30],'-lə','','suffix perfect','Source identifies perfect aspect; its form coincides with the past marker.'),
(157,[32],'rəhə-','','stem auxiliary pret','Past perfect construction.'),
(157,[32],'baṭə-','','stem auxiliary non-past','Non-past perfect construction.'),
(158,[28],'-ø','','impv','Explicit zero-marker notation, audit-only.'),
(158,[28],'-uk','','suffix impv','Non-honorific imperative marker in Table 6.1.'),
(158,[31],'-o','','suffix impv honorific','Medium-grade honorific imperative in Table 6.1.'),
(158,[34],'-səb','','suffix impv pl','Table 6.1 labels plural imperative marking optional.'),
(158,[37],'-ike həwəkʰai','','impv high-honorific','Table 6.1 gives this whole high-grade honorific construction.'),
(158,[40],'-isai','','suffix impv fut','Table 6.1 future imperative.'),
(158,[43],'-uk','','suffix impv','Neutral imperative; source distinguishes zero marking.'),
(159,[27],'-o','','suffix impv honorific','Source calls this mid-grade honorific imperative.'),
(159,[64],'-isai','','suffix impv fut','The source specifies delayed orders rather than immediate execution.'),
(160,[30],'həwəkʰai','','impv high-honorific','High-grade honorific imperative formed with an infinitive.'),
(160,[47],'dzʰun-','','prefix prohibitive','Source explicitly identifies a negative command prefix.'),
(161,[5],'-əu','','suffix','Hortative with consonant-final stems; optional plural marking.'),
(161,[5],'-əũ','','suffix','Hortative with consonant-final stems; optional plural marking.'),
(161,[6],'-u','','suffix','Hortative with vowel-final stems.'),
(161,[56],'-as','','suffix','Desiderative; source describes desire of subject/speaker.'),
(162,[37],'laɡ-','feel','verb stem','Source says this may follow the desiderative marker with no meaning difference.'),
(162,[41],'-us','','suffix opt','Source describes the wish of the speaker.'),
(163,[12],'laɡ-','feel','verb stem','Source permits this after desideratives but not optatives.'),
(163,[18],'-ne','','suffix cond','Source says this is the predominant conditional marker.'),
(163,[18],'-le','','suffix cond','Source describes a single observed -n-final verb, with -ne also accepted by consultants.'),
(163,[18],'-ne','','suffix cond','Second occurrence in the same line states predominance.'),
(163,[19],'-le','','suffix cond','Observed -n-final context.'),
(163,[20],'-ne','','suffix cond','Consultants judge this grammatical in the same context.'),
(163,[21],'-ne','','suffix cond','Source specifies usual -l-final context.'),
(163,[22],'par','if','conj cond','Source identifies conditional particles/subordinating conjunctions.'),
(163,[22],'sit','if','conj cond','Source identifies conditional particles/subordinating conjunctions.'),
(164,[35],'bʰəine','','verb cond','Source describes this following past tense in hypothetical protasis.'),
(164,[36],'-larə','','suffix cond','Hypothetical apodosis, with person and number agreement.'),
(164,[36],'laharə','','affix cond','Second printed alternative lacks initial hyphen.'),
(164,[84],'-larə','','suffix cond','Hypothetical use is always followed by agreement features.'),
(164,[84],'-larə','','suffix prospective','Second same-line occurrence contrasts prospective use without following agreement.'),
(165,[29],'-nei','','suffix','Source identifies strong obligation.'),
(165,[29,30],'pərit','happen-3SG-NPST','verb','With an infinitive, expresses weak obligation according to the author.'),
(165,[31],'-nei','','suffix','Source identifies a non-finite form expressing strong obligation.'),
(166,[63],'-bə','','suffix modal','Epistemic possibility; event has not yet occurred but may occur.'),
(166,[64],'-bo','','suffix modal','Epistemic possibility.'),
(166,[64],'-bi','','suffix modal','Epistemic possibility.'),
(167,[25],'həbai','probably','adv modal','Independent epistemic adverb; source gives həbai/həbʰai and hətahai/həbʰahai.'),
(167,[25],'həbʰai','probably','adv modal','Independent epistemic adverb.'),
(167,[25],'hətahai','probably','adv modal','Independent epistemic adverb.'),
(167,[25],'həbʰahai','probably','adv modal','Independent epistemic adverb.'),
(168,[55,56],'-inei','','suffix','Source identifies mirativity, specifically unexpected results.'),
(169,[40],'-hun','they say','suffix','Hearsay evidential distinguishes first-hand knowledge from second-hand information.'),
(169,[76],'-hun','','suffix','Source footnote explicitly equates its shape with the nominal indefinite marker; do not infer identical function.'),
(170,[5],'-les','','suffix','Frustrative: undesired or unexpected event result; occurs only with past tense.'),
(170,[47],'-les','','suffix','Source explicitly says this marker cannot be further segmented despite resemblance to -l and -s.'),
(170,[48],'-l','','suffix','Comparison within the author’s warning against segmenting -les.'),
(170,[48],'-s','','suffix 2sg','Comparison within the author’s warning against segmenting -les.'),
(170,[50],'-l','','suffix','Comparison within the author’s warning against segmenting -les.'),
(170,[50],'-s','','suffix 2sg','Comparison within the author’s warning against segmenting -les.'),
(170,[52],'-isin','','suffix','Dubitative section heading; uncertainty or confusion, restricted to second singular.'),
(170,[53],'-isin','','suffix','Dubitative; rare in the author’s discourse corpus.'),
(170,[57],'-isin','','suffix 2sg','Source specifies second-singular restriction and mainly interrogative occurrence.'),
(171,[74],'-isin','','suffix','In affirmative sentences the author attributes uncertainty to this form.'),
(172,[3],'-oi','','suffix manner degree','Manner/intensity section heading.'),
(172,[4],'-oi','','suffix manner degree','Few corpus occurrences; author explicitly requests further investigation.'),
(172,[56],'bosike','to sit','verb inf','Verb used for Table 6.2 dialect comparison.'),
(174,[15],'-hĩ','','suffix 1pl','Explicit Chitwan attribution.'),
(174,[16],'-hir','','suffix 1pl','Explicit Tanahun attribution; prose differs from the -ir form in the preceding example and table.'),
(174,[18],'-u','','suffix','Rare second-plural use; author contrasts Chitwan honorific use.'),
(174,[19,20],'-həsəb','','suffix pl','Second and third plural in Chitwan.'),
(174,[21],'-həsəb','','suffix pl','Source describes this instead of -at/-hat.'),
(174,[21],'-at','','suffix pl','Source comparison with -həsəb.'),
(174,[21],'-hat','','suffix pl','Source comparison with -həsəb.'),
(174,[52],'-həsəb','','suffix pl','Footnote: more frequent second and third plural marking.'),
(174,[53],'-at','','suffix pl','Consultants attribute this to older speakers.'),
(174,[53],'-hat','','suffix pl','Consultants attribute this to older speakers.'),
(174,[53],'-at','','suffix pl','Second occurrence in the same footnote.'),
(174,[53],'-həsəb','','suffix pl','Consultants attribute this to younger speakers.'),
(174,[54],'-at','','suffix pl','Older-generation use; source says young speakers use it honorifically.'),
(174,[54],'-hat','','suffix pl','Older-generation use; source says young speakers use it honorifically.'),
(175,[6],'-hau','','suffix 2sg honorific','Source describes second-person mid-level honorific agreement.'),
(175,[6],'-u','','suffix 2sg honorific','Source groups -u/hau in the same context.'),
(175,[6],'hau','','affix 2sg honorific','Second alternative lacks printed initial hyphen.'),
(175,[7],'-hausəb','','suffix 2pl honorific','Source describes second-person plural honorific use.'),
(175,[10],'-hat','','suffix 3pl','Author explicitly notes inconsistent third-plural use and distinguishes examples52 from honorific examples53.'),
(175,[57],'-u','','suffix 2pl','Kotapish and Kotapish (1975) attribution; historical/Tanahun comparison, not default Chitwan.'),
(175,[57],'hau','','affix 2pl','Kotapish and Kotapish (1975) attribution; second printed alternative.'),
(175,[59],'-ir','','suffix','Author says pronominal suffix less common in Chitwan.'),
(175,[59],'-u','','suffix','Author says pronominal suffix less common in Chitwan.'),
(175,[59],'-tə','','suffix','Author says pronominal suffix less common in Chitwan.'),
(175,[61],'-tahəsəb','','suffix pl','Som Lal Darai and Cham Narayan Darai use this interchangeably with -hat for second and third plural.'),
(175,[61],'-hat','','suffix pl','Som Lal Darai and Cham Narayan Darai use this interchangeably with -tahəsəb for second and third plural.'),
(175,[61,62],'-t','','suffix honorific','Source specifies third-person honorific coding.'),
(175,[61,62],'-hat','','suffix honorific','Source specifies third-person honorific coding.'),
(177,[26],'kəhələ','tell-PST','verb','Source corpus discussion of past-tense plural rarity.'),
(177,[27],'sutlə','sleep-PST','verb','Source corpus discussion of past-tense plural rarity.'),
(177,[29],'-lə','','suffix pret','Contrasted with the feminine inflection.'),
(177,[29],'-l-i','PST-F','suffix pret f','Explicit feminine-subject inflection.'),
(179,[6],'aṭə-','','stem copula auxiliary','Genitive modifier triggers agreement in these copular/auxiliary constructions.'),
(179,[6],'həkʰə-','','stem copula auxiliary','Genitive modifier triggers agreement in these copular/auxiliary constructions.'),
(179,[7],'aṭə-','','stem copula','Source refers to the Appendix3e paradigm.'),
(179,[7],'rəhə-','','stem copula','Source refers to the Appendix3e paradigm.'),
(180,[39],'dadzu','brother','noun','Source explicitly contrasts the head noun with the genitive agreement controller.'),
(180,[40],'merə','my','pron gen 1sg','Source identifies this possessor as the agreement controller.'),
(180,[41],'dadzu','brother','noun','Second explicit contrast in prose.'),
(180,[41],'merə','my','pron gen 1sg','Second explicit contrast in prose.'),
(182,[4],'-r','','suffix','Author describes pronominal agreement with the referentially specific argument.'),
(182,[4],'dzoɡi','hermit','noun','Author contrasts this subject with the agreement controller.'),
(183,[77],'de-','give','stem verb','Source discusses ditransitive subject versus object agreement.'),
(184,[39],'de','give','verb','Source reports about80 occurrences, with object coding at only3 places in its corpus.'),
(184,[40],'an','bring','verb','Source reports about40 occurrences, with object agreement only2 times.'),
(184,[42],'mar','kill','verb','Source reports subject and object coding only2 times.'),
(184,[46],'anhalmis','','verb','Whole form is explicitly analyzed by the author into an, -hal, -mi and -s.'),
(184,[46],'an','bring','stem verb','Source decomposition of anhalmis.'),
(184,[46],'-hal','past tense','suffix pret','Source decomposition of anhalmis.'),
(184,[46],'-mi','1SG','suffix 1sg','Subject in the source decomposition of anhalmis.'),
(184,[46],'-s','2SG','suffix 2sg','Object in the source decomposition of anhalmis.'),
(184,[48],'de','give','verb','Source introduces examples65–66.'),
(185,[50],'-u','','suffix honorific','Source explicitly attributes honorificity.'),
(185,[51],'-kan','','suffix 3sg honorific','Source explicitly attributes third-singular honorificity.'),
(185,[52],'de-','give','stem verb','Source calls this ditransitive.'),
(185,[52],'mar-','kill','stem verb','Source calls this transitive.'),
(185,[59],'-kan','','suffix','Young Darai speakers attribute third-person honorificity; the corpus also has3PL use, especially with copulas.'),
(186,[133],'-i','','epenthetic','Source says this sound intervenes between combined subject and object suffixes; preserve phonological context.'),
(187,[3],'-s','','suffix 2sg','Source distinguishes agreement marker from pronominal -r; both may occur in transitive verbs.'),
(187,[4],'-r','','suffix 2sg','Source distinguishes pronominal suffix from agreement marker -s.'),
(187,[30],'-r','2SG','suffix 2sg','Kotapish and Kotapish1975:140 reportedly exclude subject/receptor use for Tanahun; the author finds object use in his own corpus.'),
(190,[45],'-k','','suffix','Source contrasts marked and unmarked agreement in discourse context.'),
(190,[74],'-k','','suffix','Source explicitly leaves subject versus object interpretation ambiguous without discourse context.'),
(192,[13],'de-','give','stem verb','The author contrasts main-verb meaning with benefactive vector use.'),
(192,[16],'sək-','finish','stem verb','The source refers to examples73–74 here although they illustrate de; preserve the source mismatch.'),
(192,[45],'səkike','completive','verb','Explicit vector function, distinct from the lexical gloss in the table.'),
(192,[45],'deike','benefactive','verb','Explicit vector function, distinct from the lexical gloss in the table.'),
(192,[46],'sək-','finish','stem verb',''),
(193,[19],'sək-','finish','stem verb','Author contrasts lexical meaning in76 with bleached vector meaning in77.'),
(193,[21],'sək-','finish','stem verb','Source attributes complete-action meaning to the vector use.'),
(193,[22],'rəhə-','remain','stem verb','Source contrasts continuative vector use with lexical remain.'),
(194,[13],'dza','go','verb','Source attributes suddenness to vector use.'),
(195,[15],'laləi','instantaneous','verb','Source describes inceptive/immediate action and warns that resemblance to third-singular inflection is not a valid analysis.'),
(195,[16],'laɡ','apply','verb','Source compares the similar-looking form, explicitly distinguishing it from laləi.'),
(195,[69],'ɡʰal','put','verb','Source reports lexical use but no vector use in V1V2 constructions.'),
(195,[71],'hal-','put','stem verb','Author tentatively attributes use after the absolutive to Nepali influence; not an established borrowing claim.'),
(195,[71],'ɡʰal','','verb','Second explicit occurrence in the contrast with hal-.'),
(196,[45],'-ike','','suffix inf','Vector/modal auxiliaries in this subsection follow the infinitive.'),
(196,[69,70],'səkike','to finish','verb','Source identifies abilitative/capacity use.'),
(197,[35],'pərike','to happen','verb','Source attributes necessity function.'),
(197,[35],'aṭə','copula-NPST','copula verb','Source attributes necessity function.'),
(197,[60],'deike','to give','verb','Permission function.'),
(197,[60],'laɡike','to apply','verb','Beginning or inception function.'),
(197,[61],'bʰəit','','verb','Source labels this use obligation.'),
(198,[30],'-ike','','suffix inf','Source describes infinitival, purposive and procedural-instruction uses; nominal use is shown in90d.'),
(199,[57],'-i','','suffix abs','Source explicitly identifies the absolutive used frequently in compound verbs.'),
(200,[4],'-ikun','SEQ','suffix','Sequential converb.'),
(200,[4],'ikunu','SEQ','affix','Second printed alternative lacks initial hyphen.'),
(200,[4],'-te','SIM','suffix','Simultaneous converb.'),
(200,[4,5],'-tin','PROG','suffix progressive','Continuous converb.'),
(200,[30],'-te','','suffix','Simultaneous action; author also calls it a non-continuous converb.'),
(200,[55],'-te','','suffix','Source explicitly distinguishes habitual aspect and non-continuous converb functions.'),
(201,[38],'-lə','','suffix ptcp','Participle section heading.'),
(201,[39],'-lə','','suffix perfect','Source compares the perfect-shaped form used adjectivally or as a prenominal relative clause.'),
(202,[3],'-tihin','','suffix ptcp','Imperfective/present participle section heading.'),
(202,[5],'-tihin','','suffix progressive ptcp','Source says the progressive marker also forms imperfective/present participles and adverbials.'),
(202,[56],'-e','','suffix','Distinct purposive marker; not conflated with infinitive purposive use.'),
(203,[4],'-ne','','suffix cond','Explicit conditional marker.'),
(203,[20],'-larə','','suffix prospective','Non-finite attributive adjective and sometimes clause-final use.'),
(203,[21],'-lari','','suffix prospective f','Explicit feminine form.'),
]
FUNCTIONS={
 'dhakal2011:p192:ch6:pair:l29':('benefactive',[30]),
 'dhakal2011:p192:ch6:pair:l33':('completive',[34]),
 'dhakal2011:p192:ch6:pair:l39':('continuative',[40]),
 'dhakal2011:p192:ch6:pair:l42':('happened unexpectedly',[43]),
 'dhakal2011:p196:ch6:pair:l53':('abilitative',[54]),
 'dhakal2011:p196:ch6:pair:l56':('permissive (outward directed)',[57]),
 'dhakal2011:p196:ch6:pair:l59':('initiative (beginning)',[60]),
 'dhakal2011:p196:ch6:pair:l61':('obligation, compulsion',[62]),
 'dhakal2011:p196:ch6:pair:l64':('need',[65]),
 'dhakal2011:p196:ch6:pair:l67':('appropriateness',[68]),
}
TRANSITIVE_MATRIX = [
 ('1SG',[18,19,20,21,22,23,[24,25],[26,27],[28,29]]),
 ('1PL',[32,33,[34,35],[36,37],[38,39],40,[41,42],[43,44],[45,46]]),
 ('2SG',[[49,50],[51,52],53,54,55,56,[57,58],[59,60],[61,62]]),
 ('2SG.H',[[66,67],[68,69],70,71,72,73,[74,75],[76,77],[78,79]]),
 ('2PL',[[82,83],[84,85],86,87,88,[89,90],[91,92],[93,94],[95,96]]),
 ('3SG',[[99,100],[101,102],103,104,[105,106],107,108,[109,110],[111,112]]),
 ('3PL',[[116,117],[118,119],120,121,[122,123],124,125,[126,127],[128,129]]),
]


def build():
    pages={}
    with (HERE/'whole-native-pages.jsonl').open() as handle:
        for line in handle:
            page=json.loads(line)
            if page.get('printed_page') in range(146,206):pages[page['printed_page']]=page
    def evidence(page, lines):return [pages[page]['lines'][i-1] for i in lines]
    def text(page,line):return ''.join(s['decoded'] for s in pages[page]['lines'][line-1]['spans']).strip()
    def clean(value):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',value).strip())
    result=[]
    for page,pairs in SURFACES.items():
        for form_lines,gloss_lines in pairs:
            fs=form_lines if isinstance(form_lines,list) else [form_lines]
            gs=gloss_lines if isinstance(gloss_lines,list) else [gloss_lines]
            raw_translation=' '.join(text(page,i) for i in gs)
            result.append({'source_unit_key':f'dhakal2011:p{page}:ch6:surface:l{fs[0]}',
                'printed_page':page,'pdf_page':page+22,'kind':'whole-expression','surface_lines':fs,
                'form':clean(' '.join(text(page,i) for i in fs)),
                'raw_translation':raw_translation,
                'gloss':re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+\)','',raw_translation).strip('‘’\'" '),
                'text_ids':re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+)\)',raw_translation),
                'source_surface_evidence':evidence(page,fs),'source_translation_evidence':evidence(page,gs),
                'interlinear_evidence':pages[page]['lines'][max(fs):min(gs)-1]})
    for page,pairs in PAIRS.items():
        for line,form,gloss in pairs:
            result.append({'source_unit_key':f'dhakal2011:p{page}:ch6:pair:l{line}',
                'printed_page':page,'pdf_page':page+22,'kind':'explicit-form-gloss',
                'form':clean(form),'gloss':gloss,'evidence':evidence(page,[line])})
    occurrences={}
    for subject, cells in TRANSITIVE_MATRIX:
        for obj, raw_lines in zip(['1SG','1PL','2SG','2SG.H','2PL','3SG','3SG.H','3PL','3PL.H'],cells):
            lines=raw_lines if isinstance(raw_lines,list) else [raw_lines]
            pieces=[text(186,i) for i in lines]
            control=pieces==['-'] or pieces[0]=='Only S'
            result.append({'source_unit_key':f'dhakal2011:p186:ch6:matrix:s{subject}:o{obj}',
                'printed_page':186,'pdf_page':208,'kind':'agreement-matrix-cell',
                'form':(' '.join(pieces) if control else ''.join(pieces)),
                'gloss':'','subject_label':subject,'object_label':obj,
                'evidence':evidence(186,lines),'raw_pieces':pieces,
                'disposition':'structural-control' if control else 'target-candidate'})
    for page, cells in TABLES.items():
        for line, grammar, lect in cells:
            result.append({'source_unit_key':f'dhakal2011:p{page}:ch6:table:l{line}',
                'printed_page':page,'pdf_page':page+22,'kind':'paradigm-cell','form':clean(text(page,line)),
                'gloss':'','source_grammar':grammar,'lect':lect,'evidence':evidence(page,[line]),
                'table_context':pages[page]['lines'][2:46],
                'disposition':'zero-morphology-evidence' if text(page,line)=='-ø' else 'target-candidate'})
    for page,lines,form,gloss,tags,note in PROSE:
        stem=f'dhakal2011:p{page}:ch6:prose:l{lines[0]}'
        occurrences[stem]=occurrences.get(stem,0)+1
        result.append({'source_unit_key':f'{stem}:a{occurrences[stem]}','printed_page':page,'pdf_page':page+22,
            'kind':'prose-attestation','form':clean(form),'gloss':gloss,'tags':tags.split(),
            'note':note,'source_lines':lines,'evidence':evidence(page,lines),
            'disposition':('zero-morphology-evidence' if form=='-ø' else 'phonological-context' if tags=='epenthetic' else 'target-candidate')})
    assert len({r['source_unit_key'] for r in result})==len(result)
    for row in result:row['status']='Native first reading; original visual review and grammar integration pending'
    for row in result:
        if row['source_unit_key'] in FUNCTIONS:
            function, lines=FUNCTIONS[row['source_unit_key']]
            row['source_function']=function
            row['function_evidence']=evidence(row['printed_page'],lines)
        if row['source_unit_key']=='dhakal2011:p188:ch6:surface:l9':
            row['disposition']='source-unacceptable'
            row['source_example_status']='source-unacceptable'
            row['note']='Original explicitly prefixes an asterisk and describes the constructed example as unacceptable.'
        if row['source_unit_key']=='dhakal2011:p174:ch6:surface:l11':
            row['lect']='Tanahun'
            row['witness']='Darai (1996:38), cited by Dhakal; explicitly assigned Tanahun in the following paragraph.'
        if row['printed_page']==174 and row.get('source_lines')==[16]:row['lect']='Tanahun'
        if row['printed_page']==175 and row.get('source_lines')==[57]:
            row['lect']='Tanahun'
            row['witness']='Kotapish and Kotapish (1975), cited by Dhakal; table source note on page172.'
    return result


if __name__=='__main__':
    rows=build()
    (HERE/'chapter6-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'first-reading units; native first reading covers all pages146–205; original visual review and interlinear recovery pending')
    import hashlib
    from collections import Counter
    report={'status':'All 60 Chapter 6 pages read in native extraction; original visual review and interlinear lexical recovery pending',
        'first_reading_sha256':hashlib.sha256((HERE/'chapter6-first-reading.jsonl').read_bytes()).hexdigest(),
        'pages_read':list(range(146,206)),'counts_by_kind':dict(Counter(r['kind'] for r in rows)),
        'counts_by_page':dict(sorted(Counter(r['printed_page'] for r in rows).items())),
        'no_target_pages':[205],
        'source_context':'Chapter overview and foreign/general theoretical commentary retained in whole-native-pages.jsonl; font-class letters are contextual, not independently attested morphemes.',
        'pending':['Full original-page visual comparison','Recover all aligned lexical units and explicit alternatives','Source-wide source-claim and grammar metadata review','Independent audit','Whole-thesis completion'],
        'canonical_changed':False}
    (HERE/'chapter6-first-reading-scope-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
