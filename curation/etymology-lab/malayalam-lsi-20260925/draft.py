"""Research-only LSI Malayalam residual decisions; no accepted overlay writes."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 137

proposals = {}
holds = {}

def add(i, parents, evidence, kind=None):
    assert i not in proposals and i not in holds
    proposals[i] = (parents if isinstance(parents, list) else [parents], evidence, kind)

def hold(i, reason):
    assert i not in proposals and i not in holds
    holds[i] = reason

for i in range(1, 10):
    hold(i, 'The accʰan “father” stem is in DEDR 50, but that article explicitly judges the family probably Indo-Aryan (CDIAL 1347). The immediate donor and its persistent node are unverified; case, number and numeral material do not resolve the borrowing.')
hold(10, 'peŋŋaḷ “sister” may contain the female stem in DEDR 4395, but the kinship sense and the exact formation have not been verified; DEDR 23 akka is a separate sister word.')
add(11, 'd23', 'DEDR 23 lists Tamil akkā, Kannada akka and other South Dravidian “elder sister” forms; LSI Malayalam akkă is a family-level match despite Malayalam not being separately cited in the article. The survey gloss “Sister” is broader.')
add(12, 'd399', 'DEDR 399 includes Malayalam āḷ “person, servant” and Tamil āḷ “man”; LSI uses the corresponding personal noun for “Man”.')
hold(13, 'strī “woman” is a learned Indo-Aryan-looking form; no immediate donor attestation with a persistent node was established.')
hold(14, 'bhāryyā “wife” is a learned Indo-Aryan-looking form; no immediate donor attestation with a persistent node was established.')
add(15, 'd4616', 'DEDR 4616 directly gives Malayalam makan “son” beside Tamil makaṉ; LSI magan is the corresponding survey spelling.')
add(16, 'd4616', 'DEDR 4616 contains Malayalam mōḷ “daughter” and Tamil makaḷ; the LSI magaḷ is a transparent family member despite a different Malayalam dictionary surface form.')
add(17, 'd4616', 'LSI magaḷŭ gaḷ is a plural daughter response; DEDR 4616 identifies the child/daughter stem, while the plural element remains in the source record.')
for i, parents, animal in [(18,['d399','d1711'],'horse'),(19,['d4395','d1711'],'mare'),(20,['d399','d1711'],'horses'),(21,['d4395','d1711'],'mares')]:
    add(i, parents, f'Ordered sex + animal response for “{animal}”: DEDR {parents[0][1:]} gives Malayalam āṇ “male” or peṇ “female”, and DEDR 1711(a) directly gives Malayalam kutira “horse”. LSI number marking stays in the form.')
add(22, 'd1502', 'DEDR 1502 directly gives Malayalam kāḷa “bull, bullock”; this sense excludes the homonymous blackness family.')
hold(23, 'paśŭ “cow” is a learned Indo-Aryan-looking noun; immediate donor and borrowing node are not established.')
add(24, 'd1502', 'The plural bull response contains Malayalam kāḷa “bull, bullock” in DEDR 1502; gaḷ is retained as survey number marking.')
hold(25, 'Plural paśukkaḷ inherits the unresolved paśŭ “cow” donor problem; do not assign a remote ancestor to the whole response.')
add(26, 'd3650', 'DEDR 3650 directly gives Malayalam nāy “dog”; -kkaḷ is plural marking in the LSI response.')
add(27, 'd3870', 'DEDR 3870 directly gives Malayalam paṭṭi “dog, bitch”; the LSI plural -gaḷ is grammatical.')
add(28, ['d5496','d5152'], 'LSI veḷḷāḍă combines the white stem veḷḷa (DEDR 5496(a)) and Malayalam āṭu “goat” (DEDR 5152); the survey “he goat” gloss need not make whiteness the sex marker.')
add(29, ['d4395','d5496','d5152'], 'Ordered peṇ “female” (DEDR 4395) + veḷḷa “white” (5496(a)) + āṭu “goat” (5152); the female qualification is separately represented.')
add(30, 'd5152', 'DEDR 5152 directly gives Malayalam āṭu “goat”; the LSI -gaḷ is plural marking.')
for i, parents, gloss in [(31,['d399','d4780'],'male deer'),(32,['d4395','d4780'],'female deer')]:
    add(i, parents, f'Ordered sex + deer response for “{gloss}”: DEDR {parents[0][1:]} gives the Malayalam sex word and DEDR 4780 directly gives Malayalam mān “deer”.')
add(33, 'd4780', 'DEDR 4780 directly gives Malayalam mān “deer”; -gaḷ is plural marking.')
for i in (34,35,37):
    hold(i, 'The good adjective nalla is securely in DEDR 3610, but the accompanying manuṣyan/manuṣyar or strī noun is learned Indo-Aryan-looking with no verified immediate donor node; the full multiword response remains unresolved.')
hold(36, 'DEDR 3610 gives Malayalam nalla “good” and nallavan “good man”, but a derived relation for LSI nallavaḷ needs a checked Malayalam source node for the immediate stem; the DEDR reconstruction is not that synchronic parent.')
for i in (38,39):
    hold(i, 'DEDR 2616 gives Malayalam cītta “badness”, but āŋguṭṭĭ/peŋguṭṭĭ “boy/girl” and the complete adjective+noun segmentation need further primary verification.')
add(40, 'd646', 'DEDR 646 directly gives Malayalam uyaruka “be high” and uyaram “height”; LSI uyară is the corresponding high stem.')
add(41, 'd5086', 'DEDR 5086 directly gives Malayalam mēl “what is above” and mēlē “upwards”; LSI mēl has the same spatial sense.')
add(42, 'd1619', 'DEDR 1619 directly gives Malayalam kīr̤ “below, down”; LSI kīṛ preserves the survey spelling of the retroflex liquid.')
add(43, 'd5020', 'DEDR 5020(a) gives Malayalam mumpu “front, before”; LSI munbĕ is its before-form with survey b/p spelling.')
add(44, 'd4205', 'DEDR 4205 gives Malayalam pin “behind” and pinnē “behind, after”; LSI pinnālē is the same spatial family with postpositional material.')
add(45, 'd5151', 'DEDR 5151 directly gives Malayalam yār/ār “who”; LSI ār matches the shorter interrogative form.')
add(46, 'd5151', 'DEDR 5151 directly gives Malayalam entu “what; why”; LSI endŭ is the survey voicing of the same interrogative.')
hold(47, 'endŭkoṇḍŭ “why” combines a verified interrogative (DEDR 5151) with koṇḍŭ; the latter component and compound analysis await primary verification.')
add(48, 'd5151', 'DEDR 5151 directly gives Malayalam entu “what; why”; LSI endinnŭ is an inflected why-response of that interrogative stem.')
hold(49, 'um “and” is a grammatical enclitic; no suitable verified lexical parent was identified.')
hold(50, 'eŋgilum “but” is a concessive connective; its historical components and appropriate parent relation remain unverified.')
hold(51, 'adĕ “yes” may be a deictic assent expression, but DEDR 1 only supports a demonstrative family and DEDR 333 a different assent form ām; exact parent unresolved.')
add(52, 'd2559', 'DEDR 2559 directly gives Malayalam illa “does not exist; no”; LSI illă is the corresponding negative answer.')
add(53, 'd4572', 'DEDR 4572 directly gives Malayalam pōka “to go”; LSI pō is the imperative/base form.')
add(54, 'd480', 'DEDR 480 directly gives Malayalam irikkuka “sit, remain”; LSI irĭ is an abbreviated survey imperative.')
add(55, 'd474', 'DEDR 474 gives Malayalam iru-patu “twenty”, from the two stem iru + ten; LSI irupadu is the corresponding numeral.')
add(56, 'd2826', 'DEDR 2826 directly gives Malayalam am-patu “fifty”; LSI ambadu is a survey spelling of that compound numeral.')
add(57, 'd5160', 'DEDR 5160 gives Malayalam ñān, oblique en- “I”; LSI enṛe “my” is an inflected possessive of that pronominal stem.')
add(58, 'd3647', 'DEDR 3647 directly gives Malayalam nām “we, I and you”; this is the inclusive pronoun.')
add(59, 'd5154', 'DEDR 5154 directly gives Malayalam ñā̆ṅṅaḷ “we” exclusive; the LSI generic gloss omits that distinction.')
add(60, 'd3647', 'DEDR 3647 gives Malayalam nām, oblique nam(m)- “we” inclusive; nammuḍe is its possessive response.')
add(61, 'd5154', 'DEDR 5154 gives Malayalam ñā̆ṅṅaḷ, oblique ñaṅṅaḷ- “we” exclusive; ñaŋŋaḷuḍe is the possessive response.')
add(62, 'd3684', 'DEDR 3684 gives Malayalam nī, oblique nin(n)- “you” singular; ninṛe is the corresponding singular possessive.')
add(63, 'd3688', 'DEDR 3688 directly gives Malayalam niṅṅaḷ “you” plural; LSI does not mark plurality in the English gloss.')
add(64, 'd5270', 'DEDR 5270 gives Malayalam varuka, imperative vā “come”; LSI vā is that imperative.')
add(65, 'd3675', 'DEDR 3675 gives Malayalam nilkka “stand”; LSI nil represents the same verbal stem.')
add(66, 'd2426', 'DEDR 2426 gives Malayalam cāka “die”; LSI cāg has survey voicing of the final stop.')
add(67, 'd3098', 'DEDR 3098 gives Malayalam taruka, imperative tā “give”; LSI tā is the imperative.')
add(68, 'd1041', 'DEDR 1041 gives Malayalam ōṭuka “run”; LSI ōḍ is the verbal stem.')

subjects = {'ñān':'d5160', 'nī':'d3684', 'avan':'d1', 'ñaŋŋaḷ':'d5154', 'niŋŋaḷ':'d3688', 'avar':'d1'}
for i in list(range(69,75))+list(range(75,80)):
    word=records[i-1]['Form'].split()[0]
    assert word in subjects,(i,word)
    add(i,[subjects[word],'d333'],f'Ordered pronoun + finite “be” response: DEDR {subjects[word][1:]} attests Malayalam {word} or its matching pronoun, and DEDR 333 gives Malayalam ākuka “be, become”. Tense/person marking is retained in the LSI form.')
hold(80, 'The source has avar āyĭavarŭ iddarŭ in one “They were” response, apparently conflating two predicate forms. Do not collapse the entire elicitation into a single two-component analysis.')
add(81,'d77','DEDR 77 directly gives Malayalam aṭikka “beat, strike”; LSI aḍi is the beat stem, distinct from DEDR 72 “foot” and 73 “sweep”.')
for i in range(82,100):
    word=records[i-1]['Form'].split()[0]
    assert word in subjects,(i,word)
    add(i,[subjects[word],'d77'],f'Ordered pronoun + beat response: DEDR {subjects[word][1:]} gives the Malayalam {word} pronoun and DEDR 77 directly gives aṭikka “beat, strike”. Present, past or future morphology stays in the LSI record.')
for i in range(100,112):
    if i==103:
        hold(i,'The source says “We go” but records only ñaŋŋaḷ, with no go verb; retain the incomplete response rather than adding an unattested verbal component.')
        continue
    word=records[i-1]['Form'].split()[0]
    assert word in subjects,(i,word)
    add(i,[subjects[word],'d4572'],f'Ordered pronoun + go response: DEDR {subjects[word][1:]} gives the Malayalam {word} pronoun and DEDR 4572 directly gives pōka “go”. The finite ending is grammatical.')
add(112,'d3688','DEDR 3688 gives Malayalam niṅṅaḷ, oblique niṅṅaḷ- “you” plural; niŋŋaḷuḍe is its possessive.')
add(113,'d3688','DEDR 3688 gives Malayalam niṅṅaḷ, oblique niṅṅaḷ- “you” plural; niŋŋaḍe is a reduced possessive survey variant.')
add(114,'d1','DEDR 1 directly gives Malayalam avan “that man, he” in its demonstrative-pronoun article.')
add(115,'d1','DEDR 1 gives Malayalam avan “that man, he”; avanṛe is its possessive response.')
add(116,'d1','DEDR 1 directly gives Malayalam avar “those persons, they” in its demonstrative-pronoun article.')
add(117,'d1','DEDR 1 gives Malayalam avar “those persons, they”; avaruḍe is its possessive response.')
add(118,['d3103','d4921'],'LSI talamuḍi “hair” contains Malayalam tala “head” (DEDR 3103) and muṭi “hair of head” (DEDR 4921), preserved in source order.')
add(119,'d4955','DEDR 4955 directly gives Malayalam mutuku “back, spine”; LSI mudugă is its survey rendering.')
add(120,'d486','DEDR 486 directly gives Malayalam irimpu “iron”; LSI irimbă reflects the same iron noun with survey voicing.')
hold(121,'kṛṣīvalan “cultivator” is a learned Indo-Aryan-looking occupational noun with no verified immediate donor node.')
add(122,'d5152','DEDR 5152 explicitly gives Malayalam āṭṭukāran “shepherd” beside āṭu “goat”; LSI ātṭugāran is a survey voicing of that agent noun.')
hold(123,'deivam “god” is an Indo-Aryan-looking religious noun; the immediate donor and relation are not verified.')
hold(124,'piśācă “devil” is an Indo-Aryan-looking religious noun; the immediate donor and relation are not verified.')
add(125,'d5503','DEDR 5503 directly gives Malayalam veḷḷam “water”; unlike the white adjective, this is the water article.')
add(126,'d1655','DEDR 1655 directly gives Malayalam kuṭi “house, hut, family”; this sense excludes DEDR 1654 “drink”.')
add(127,'d1711','DEDR 1711(a) directly gives Malayalam kutira “horse”; LSI kudiră is the survey spelling.')
add(128,'d1364','DEDR 1364 directly gives Malayalam kar̤uta “ass”; LSI kaṛudă is its survey spelling. The article compares Sanskrit gardabha-, so the proposal is a family-level identification only, without claiming deeper inheritance.')
hold(129,'oṭṭagam “camel” is an Indo-Aryan-looking animal noun; immediate donor and exact borrowing history are unverified.')
hold(130,'The repeated paśŭ “cow” record has the same unresolved immediate donor as the earlier cow response.')
add(131,'d4355','DEDR 4355 directly gives Malayalam pūcca “cat”; LSI matches that noun.')
for i,animal in [(132,'paksĭ “bird”'),(133,'pūvaŋgōṛĭ “cock”'),(134,'bāttŭ “duck”')]:
    hold(i,f'{animal} has no verified complete lexical family or immediate donor for this LSI spelling; avoid forcing an animal-name comparison.')
for i,body in [(135,'sūryan “sun”'),(136,'candran “moon”'),(137,'nakṣatram “star”')]:
    hold(i,f'{body} is a learned Indo-Aryan-looking astral noun; immediate donor node and borrowing path are unverified.')

assert set(proposals).isdisjoint(holds)
assert set(proposals)|set(holds)==set(range(1,138))
accepted=[]
for i,(parents,evidence,kind) in sorted(proposals.items()):
    record=records[i-1]
    row=dict(record=record,parent=parents[0],kind=kind or ('component' if len(parents)>1 else 'reflex'),
             citation=record['Source']+''.join(f';dedr[{p[1:]}]' for p in parents),
             evidence=evidence,comparison='Full DEDR article and exact LSI source record checked.',status='pending-review')
    if len(parents)>1:row['components']=parents
    accepted.append(row)
held=[dict(record=records[i-1],candidates=[],reason=reason) for i,reason in sorted(holds.items())]
(P/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=accepted,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text('[]\n')
ids={p for ps,_,_ in proposals.values() for p in ps}|{'d50','d72','d73','d1654','d2616','d3610'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for r in csv.DictReader(stream):
        if r['Entry_ID'] in ids:articles[r['Entry_ID']]=r
assert ids<=articles.keys(),ids-articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print(len(accepted),'proposed records',sum(len(x.get('components',[x['parent']])) for x in accepted),'rows',len(held),'holds')
