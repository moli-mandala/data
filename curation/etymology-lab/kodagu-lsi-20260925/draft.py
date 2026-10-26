"""Prepare a research checkpoint for LSI Kodagu; no accepted graph writes."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())

# Keys are one-based positions in the fixed 92-record residual inventory.
# Whole DEDR article nodes are the proposed parents; an attested Kodagu form
# need not match the article's first reconstructed/head form exactly.
proposals = {
    1: ('d2485', 'DEDR 2485 explicitly records Kodagu a·rï “six”, beside Tamil āṟu and Kannada āru. LSI āry reflects its rounded-vowel transcription; the r/ry rendering is retained.'),
    2: ('d910', 'DEDR 910 explicitly records Kodagu ë·ḷï “seven”, alongside Tamil ēr̤u, Malayalam ēr̤u and Kannada ēḷu. LSI yēḷy differs in initial and final-vowel notation.'),
    3: ('d784', 'DEDR 784 explicitly records Kodagu ëṭṭï “eight”, alongside Tamil eṭṭu, Kota eṭ and Kannada eṇṭu. LSI yeṭṭy keeps its own initial glide and final rounded-vowel notation.'),
    4: ('d3918', 'DEDR 3918 explicitly records Kodagu pattï “ten”, and its compound numeral stems, with Tamil and Kannada pattu. LSI patty is the same cited numeral in survey spelling.'),
    5: ('d474', 'DEDR 474 explicitly records Kodagu daṇḍï “two things”, with Tamil iraṇṭu and Kannada eraḍu. LSI danḍy is recorded as the survey numeral “Two”; the article has the entire two-family.'),
    6: ('d5052', 'DEDR 5052 explicitly records Kodagu mu·ndï “three”, with Tamil mūṉṟu and Kannada mūru. LSI mūndy preserves a medial nasal-stop cluster and differs in final-vowel notation.'),
    7: ('d3655', 'DEDR 3655 explicitly records Kodagu na·lï “four”, beside Tamil nāṉku and Kannada nālku. LSI nāly is its survey rendering.'),
    9: ('d4616', 'DEDR 4616 directly gives Kodagu mo·va “daughter”, beside the daughter/child forms of Tamil and Kannada. LSI mōva has the same form and sense.'),
    10: ('d4616', 'LSI mōḷia is the plural “Daughters”; DEDR 4616 covers the daughter/child family and explicitly gives Kodagu mo·va “daughter”. The exact plural mōḷia is not quoted there, so the form is provisionally a family-level comparison.'),
    11: ('d1711', 'DEDR 1711(a) directly gives Kodagu kudïre “horse”, with Tamil kutirai and Kannada kudire. The LSI restriction “Horse (male)” is an elicitation qualifier, not a separate stem.'),
    14: ('d3610', 'DEDR 3610 directly gives Kodagu nallë “good”, beside Tamil nalla and Kannada nalla. LSI nallad is a nominalized “good man”; its final -d and human gloss are retained as qualifications.'),
    16: ('d5151', 'DEDR 5151 explicitly gives Kodagu a·rï “who” and the genitive a·ḍa in the interrogative article. LSI āry has the same phonological stem as its separate “Six” homograph, but the interrogative gloss identifies this family.'),
    17: ('d5151', 'DEDR 5151 contains the Dravidian e-/ē- interrogative base, including Kannada ē “what”; Kodagu ennï “what” is quoted there. LSI yē is shorter than the quoted Kodagu form, so this links the broad interrogative family with the exact survey spelling retained.'),
    18: ('d333', 'DEDR 333 explicitly gives Kodagu a·ku “yes, all right” within the a·g- “become” article. LSI akku “Yes” may geminate the stop or reflect survey notation; the direct semantic match is explicit.'),
    19: ('d234', 'DEDR 234 explicitly gives Kodagu alla “to be not so-and-so” for all persons and numbers, beside Malayalam alla and Kannada alla. LSI uses the same negative form as the response “No”.'),
    20: ('d4572', 'DEDR 4572 directly gives Kodagu po·- “go”, including stems po·p-, po·c-, po·k- and po·y-. LSI pō is the bare elicited verb.'),
    22: ('d474', 'DEDR 474 directly prints Kodagu iru-vadï “twenty” in the two-numeral family, alongside its two-stem daṇḍï. LSI iruvady is the same compound numeral in survey notation.'),
    23: ('d2826', 'DEDR 2826 directly prints Kodagu aym-badï “fifty” in the five-numeral family; DEDR 3918 separately describes -badï as ten in compounds. LSI eimbady matches the whole attested fifty formation.'),
    24: ('d3729', 'DEDR 3729 explicitly gives Kodagu nu·rï “100”, with Tamil nūṟu and Kannada nūṟu. LSI nūry is its survey rendering of the same numeral.'),
    25: ('d5160', 'DEDR 5160 directly gives Kodagu na·nï “I” and its oblique en-/nan-/na·-, beside Tamil nāṉ and Kannada nān. LSI nāny matches the long nominative.'),
    26: ('d5160', 'DEDR 5160 directly gives the short Kodagu na· “I”, alongside na·nï and the oblique en-/nan-/na·-. LSI nā is the corresponding short form.'),
    27: ('d5160', 'DEDR 5160 gives Kodagu first-person oblique en-/nan-/na·-; LSI ennaḍa “My” visibly uses the en(n)- stem with a possessive ending. The complete genitive is not quoted in that article.'),
    29: ('d5154', 'DEDR 5154 directly gives Kodagu eŋga “we”, with oblique eŋga-, beside Tamil yām and Malayalam ñaṅṅaḷ. LSI eŋga has the same form and gloss.'),
    30: ('d5154', 'DEDR 5154 directly gives Kodagu naŋga “we”, with oblique naŋga-. LSI naŋga has the same form and gloss.'),
    31: ('d5154', 'DEDR 5154 quotes Kodagu eŋga and its oblique eŋga- “we”; LSI eŋgaḍa “Our” preserves that first-person plural base with a possessive ending. The full possessive is not printed in the article.'),
    32: ('d5154', 'DEDR 5154 quotes Kodagu naŋga and its oblique naŋga- “we”; LSI naŋgaḍa “Our” preserves that alternate first-person plural base with a possessive ending.'),
    33: ('d3684', 'DEDR 3684 gives Kodagu ni·nï/ni· “thou” with oblique nin-/ni·-. LSI ninaḍa “Thy” uses the nin- stem plus a possessive ending; the full possessive is not separately quoted there.'),
    34: ('d3684', 'DEDR 3684 gives Kodagu second-person singular ni· with short oblique ni·-. LSI niḍa “Thy” plausibly uses this reduced stem plus a possessive ending; keep its distinct spelling alongside ninaḍa.'),
    35: ('d3688', 'DEDR 3688 explicitly gives Kodagu niŋga “you” with oblique niŋga-, distinct from singular ni·nï in DEDR 3684. LSI niŋga is the same plural form.'),
    36: ('d5270', 'DEDR 5270 directly gives Kodagu bar- “come”, with imperative/alternate stem ba· and past band-. LSI bā is the bare elicited imperative stem.'),
    38: ('d3098', 'DEDR 3098 directly gives Kodagu tar- “give to first or second person” with stem ta·, beside Tamil taru and Kannada taru. LSI tā is this bare stem; the recipient restriction belongs to the dictionary paradigm.'),
    90: ('d1711', 'DEDR 1711(a) directly gives Kodagu kudïre “horse”, with Tamil kutirai and Kannada kudire. This is the separate LSI “Horse” survey record, not a duplicate of its “Horse (male)” record.'),
}

holds = {
    8: 'LSI mānus “Man” resembles an Indo-Aryan mānuṣa-type word, but the immediate donor and exact source form are not established; DEDR 399 Kodagu a·ḷï/ a·ṇï is a different stem.',
    12: 'LSI payyu “Cow” occurs incidentally in DEDR 2659 cuḷḷi payyu “spotted cow”, but that article etymologizes “spot”; it does not establish the ancestry or immediate donor of payyu.',
    13: 'DEDR 2165 gives Kodagu kori “sheep” and Tulu kuri “sheep”; LSI kuri is “He goat”. Form and animal sense are close but not exact, and the possible borrowed/immediate donor path is unresolved.',
    15: 'LSI kuttaḍ “Bad boy” is a nominal phrase/derived form; no matching Kodagu DEDR stem or safe component analysis was established.',
    21: 'LSI kullona “Sit” has no secure match in the Kodagu DEDR sit/descend entries; DEDR 502 explicitly questions a different aḷa- “sit down” comparison.',
    28: 'LSI ēḍa “My” may be a reduced first-person possessive, but its vowel and absence of n do not match the Kodagu en-/nan-/na·- obliques quoted in DEDR 5160 closely enough for a confident assignment.',
    37: 'LSI entuna “Stand” differs from DEDR 3675 Kodagu nill- “stand”; no matching en-/entu- stem or derivation has been established.',
    39: 'LSI pār “Run” has no secure Kodagu DEDR match; DEDR 1041 Kodagu o·ḍ- “run” is a different stem and DEDR 3949 para- is “crawl/spread”.',
    89: 'LSI orama “Hair” has no securely matched Kodagu DEDR head or immediate donor; DEDR 3103 talami “hair” and DEDR 2655 jïṭṭï “tuft” are different stems.',
    91: 'LSI payyu “Cow” is a separate survey record from the male/elicitation section. The incidental DEDR 2659 phrase cuḷḷi payyu does not etymologize payyu itself.',
    92: 'LSI pakki “Bird” appears incidentally in DEDR 1608 and 5370 compound bird names, but neither article etymologizes the simplex or proves an immediate borrowing donor.',
}

assert set(proposals).isdisjoint(holds)
assert len(proposals) + len(holds) == 43
accepted=[]
for i,(parent,evidence) in proposals.items():
    record=records[i-1]
    accepted.append(dict(record=record,parent=parent,kind='reflex',citation=record['Source']+';dedr['+parent[1:]+']',evidence=evidence,comparison='Full DEDR article compared with exact LSI survey record and transcription.',status='pending-review'))
held=[dict(record=records[i-1],candidates=[],reason=reason) for i,reason in holds.items()]
decisions=dict(status='pending-review',accepted=accepted,held=held)
(P/'proposed-decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
all_articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in {v[0] for v in proposals.values()} | {'d2659','d2165','d502','d3675','d3949','d3103','d1608','d5370','d399'}:
            all_articles[row['Entry_ID']]=row
assert {v[0] for v in proposals.values()} <= all_articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(all_articles,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(accepted),'held',len(held),'unexamined',len(records)-len(accepted)-len(held),'articles',len(all_articles))
