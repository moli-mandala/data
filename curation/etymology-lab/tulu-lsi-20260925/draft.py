"""First 50 LSI Tulu residual records: pending-review research only."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
assert len(records)==137

father='DEDR 183 quotes Tulu amme “father” alongside amma “mother, lady”. LSI uses the amma-/ammāḍḷ- stem in an explicitly father-marked case paradigm; this is a family-level proposal preserving each inflected record and its source gloss, not an assertion that standalone amma always means father.'
proposals={
    1:(['d2826'],'DEDR 2826 directly gives Tulu ainů “five”, beside Tamil aintu and Kannada ayidu. LSI einy has an initial-vowel and final-rounded-vowel survey rendering.'),
    2:(['d910'],'DEDR 910 directly gives Tulu ēḷů “seven”, beside Tamil ēr̤u and Kannada ēḷu. LSI yēḷy adds an initial glide and retains its final rounded-vowel rendering.'),
    3:(['d784'],'DEDR 784 directly gives Tulu eṇma/eṇůma “eight”, beside Tamil eṭṭu and Kannada eṇṭu. LSI yeṇma is the same Tulu variant with an initial glide.'),
    4:(['d3918'],'DEDR 3918 directly gives Tulu pattů “ten”, beside Tamil and Kannada pattu. LSI patty is its survey transcription.'),
    5:(['d474'],'DEDR 474 directly gives Tulu raḍḍů “two things” in the two-numeral article, beside Kannada eraḍu and Telugu reṇḍu. LSI raḍḍy is the same numeral with final-vowel notation.'),
    6:(['d3655'],'DEDR 3655 directly gives Tulu nālů “four”, beside Tamil nāṉku and Kannada nālku. LSI nāly is its survey transcription.'),
    7:(['d183'],father),8:(['d183'],father),9:(['d183'],father),10:(['d183'],father),11:(['d183'],father),12:(['d183'],father),13:(['d183'],father),
    14:(['d5054'],'DEDR 5054 directly gives Tulu megge/megye/meggi “younger brother”, with Koraga megi/megge. LSI meggʸe “Brother” leaves the age qualification unstated; the phonetic family match is close.'),
    15:(['d399'],'DEDR 399 directly gives Tulu āḷů “person, labourer, messenger”, also āṇů “male”. LSI āly “Man” matches the person stem with a narrower male gloss.'),
    19:(['d4616'],'DEDR 4616 directly gives Tulu mage (plural magāḍḷu) “son”, beside Tamil makaṉ and Malayalam makaṉ. LSI mage has the same form and gloss.'),
    20:(['d4616'],'DEDR 4616 directly gives Tulu magaḷu (plural magaḷāḍḷu) “daughter”, beside Tamil makaḷ. LSI magaḷy is the same source form with final-vowel notation.'),
    21:(['d399','d1711'],'The source response āṇy kudure “Horse (male)” is explicitly two words. DEDR 399 gives Tulu āṇů “male” and DEDR 1711(a) Tulu kudurè “horse”; ordered components preserve both and the elicited sex restriction.'),
    22:(['d4395','d1711'],'The source response poṇṇu kudure “Mare” is two words. DEDR 4395 gives Tulu poṇṇu “girl, female” and DEDR 1711(a) kudurè “horse”; ordered components preserve both lexical parts.'),
    23:(['d4593'],'DEDR 4593 directly gives Tulu bōri “bull, ox”, beside Kannada hōri “bull calf”. LSI bōri is the same form and gloss.'),
    24:(['d399','d3650'],'The source response āṇy nāyi “Dog (male)” is two words. DEDR 399 quotes Tulu āṇů “male” and DEDR 3650 quotes Tulu nāyi “dog”; ordered components retain the elicited qualifier.'),
    25:(['d4466'],'DEDR 4466 directly gives Tulu boggi “bitch”, alongside bogge “dog”. LSI boggi is exact in form and gloss.'),
    26:(['d4395','d3650'],'The source response poṇṇu nāyi “Bitch” is two words. DEDR 4395 quotes poṇṇu “female” and DEDR 3650 nāyi “dog”; both lexical components are retained rather than linking the phrase only to dog.'),
    27:(['d399','d5152'],'The source response āŋy yēḍy “He goat” has male and goat words. DEDR 399 gives Tulu āṇů “male” and DEDR 5152 gives Tulu ēḍů “goat”; LSI āŋy has a velar rather than retroflex nasal, so the first component is provisional.'),
    28:(['d4395','d5152'],'The source response poṇṇu yēḍy “Female goat” has two attested components: DEDR 4395 Tulu poṇṇu “female” plus DEDR 5152 ēḍů “goat”. The survey initial glide is retained.'),
    34:(['d3869','d399'],'The source paḍike āṇy “Bad boy” combines DEDR 3869 Tulu paḍikè “bad, corrupt” with DEDR 399 āṇů “male; boy” (Bright and Ramanujan). Ordered components preserve the two-word response.'),
    35:(['d3869','d4395'],'The source paḍike poṇṇu “Bad girl” combines DEDR 3869 Tulu paḍikè “bad, corrupt” and DEDR 4395 poṇṇu “girl, female”. Ordered components preserve the two-word response.'),
    37:(['d4841'],'DEDR 4841 directly gives Tulu mittů “above, upon, over”, beside Tamil mī/mītu “top, above”. LSI mitty “Up” is the adverbial survey form.'),
    38:(['d2584'],'DEDR 2584 directly gives Tulu tirtů “down, beneath, below”, beside Gondi sir “beneath”. LSI tirty is the same Tulu form with final-vowel notation.'),
    42:(['d4205'],'DEDR 4205 directly gives Tulu piravu “behind, in the rear”, beside pira/pirapa “after, back”. LSI piravu has identical form and sense.'),
    43:(['d5151'],'DEDR 5151 directly gives Tulu ērů “who” in the broad interrogative article. LSI yēry has initial glide and rounded final-vowel notation; keep this distinct from its “what” and “why” records.'),
    44:(['d5151'],'DEDR 5151 directly gives Tulu dānè “what”, alongside dā/dāva. LSI dāne is the same interrogative with simplified accent notation.'),
    45:(['d5151'],'DEDR 5151 directly gives Tulu dāye “why”. LSI dāye has the same form and sense.'),
    50:(['d2559'],'DEDR 2559 directly gives Tulu ijji “not to exist, not to be”, beside B-K iddi. LSI uses the same negative form as its response “No”.'),
}

holds={
    16:'LSI naramāni “Man” has a Sanskrit-looking nara- element, but the second element and immediate donor are unresolved. DEDR 399 Tulu āḷů/āṇů is a separate stem.',
    17:'LSI strī “Woman” is a Sanskrit-looking loanword; no verified immediate donor attestation has been selected. DEDR 4395 Tulu poṇṇu is a different native stem.',
    18:'LSI bāle “Child” may be an Indo-Aryan bāla-type word, but the immediate donor has not been verified; fuzzy DEDR hits are unrelated fish and verb forms.',
    29:'The male adjective āṇy is supported by DEDR 399, but the second word ure “deer” has no verified Tulu DEDR parent. Hold the full two-word response.',
    30:'The female adjective poṇṇu is supported by DEDR 4395, but the second word ure “deer” has no verified Tulu DEDR parent. Hold the full two-word response.',
    31:'DEDR 848 supports yeḍḍe “good”, but naramāni “man” lacks a verified parent or immediate donor. Hold the entire response rather than assigning only its adjective.',
    32:'DEDR 848 supports yeḍḍe “good”, but inflected naramānyery “men” retains the unresolved naramāni stem. Hold the entire response.',
    33:'DEDR 848 supports yeḍḍe “good”, but strī “woman” has an unresolved immediate donor. Hold the entire two-word response.',
    36:'LSI nīlya “High” has no secure Tulu DEDR match; DEDR 4841 mittara “high” is a different stem.',
    39:'LSI keitaḷy “Near” has no securely matched Tulu DEDR source; superficial kai/keitaḷ comparisons do not establish the whole adverb.',
    40:'LSI akara “Far” has no securely matched Tulu DEDR source; the homepage fuzzy matches are semantically unrelated.',
    41:'LSI dumbu “Before” does not match the DEDR 5020 Tulu mundè “before” without an unsupported consonant change; retain source spelling and hold.',
    46:'LSI bokka “And” has no securely matched DEDR connective; homepage bokkaṇa means “bag, pocket”.',
    47:'LSI enclitic lā “And” is not verified as a standalone lexical stem or DEDR node; linking a clitic by near spelling would obscure its function.',
    48:'LSI āṇḍalā “But” has no verified DEDR connective or safe segmentation; retain it as an unresolved particle.',
    49:'LSI andy “Yes” is not the DEDR 333 Tulu āvu “yes”; no matching article has been verified.',
}
assert set(proposals).isdisjoint(holds)
assert len(proposals)+len(holds)==50
dec=dict(status='pending-review',accepted=[],held=[])
for i,(parents,evidence) in proposals.items():
    r=records[i-1];item=dict(record=r,parent=parents[0],kind='component' if len(parents)>1 else 'reflex',citation=r['Source']+';'+ ';'.join('dedr['+p[1:]+']' for p in parents),evidence=evidence,comparison='Full DEDR article and exact LSI source row compared; source spelling preserved.',status='pending-review')
    if len(parents)>1:item['components']=parents
    dec['accepted'].append(item)
dec['held']=[dict(record=records[i-1],candidates=[],reason=reason) for i,reason in sorted(holds.items())]
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(dec['held'],ensure_ascii=False,indent=2)+'\n')
needed={p for parents,_ in proposals.values() for p in parents} | {'d848','d5020','d333'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(dec['accepted']),'rows',sum(len(x.get('components',[x['parent']])) for x in dec['accepted']),'held',len(dec['held']),'unexamined',len(records)-len(dec['accepted'])-len(dec['held']))
