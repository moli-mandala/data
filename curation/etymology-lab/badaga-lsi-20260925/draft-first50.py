"""Research-only first checkpoint for LSI Badaga; never writes accepted sidecars."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 141

# One-based positions in the fixed residual inventory. DEDR IDs label whole
# articles, not just their displayed reconstructed heads.
proposals = {
    1: (['d2826'], 'DEDR 2826 gives Kannada ay(i)du “five”, beside its ay-vattu “fifty”; LSI eidu preserves the same numeral with survey vowel notation.'),
    2: (['d910'], 'DEDR 910 gives Kannada ēr̤u “seven” and Tulu ēḷů; LSI yēllu has an initial glide and ll for the lateral.'),
    3: (['d474'], 'DEDR 474 gives Kannada eraḍu “two”; LSI yeraḍu adds an initial glide to the same numeral.'),
    4: (['d156'], 'DEDR 156(a) directly gives Kannada appa “father”; the linked Badaga dictionary also has appa with the father sense.'),
    5: (['d3067'], 'DEDR 3067 directly gives Kannada tande “father”; the linked Badaga dictionary gives the identical form and sense.'),
    6: (['d3067'], 'LSI tandea is “of a father”; the father stem tande is in DEDR 3067, while -a is a case ending, not a separate lexical etymon.'),
    7: (['d3067'], 'LSI tande ga is “to a father”; tande is the DEDR 3067 father stem and ga is an unanalysed case particle.'),
    8: (['d3067'], 'LSI tande inda is “from a father”; DEDR 3067 identifies tande, while inda belongs to the inflected survey response.'),
    9: (['d3067'], 'LSI tande glo is plural “fathers”; DEDR 3067 identifies the father stem, with plural glo retained in the source form.'),
    10: (['d273'], 'DEDR 273 directly gives Kannada avve “mother”; the linked Badaga dictionary has the same maternal term.'),
    11: (['d364'], 'DEDR 364 gives Kannada tāy/tāyi “mother” and Tamil tāy; LSI tai is the survey spelling. The Badaga dictionary also has tāi “mother”, distinct from tai “wet”.'),
    12: (['d131'], 'DEDR 131 gives Kannada aṇṇa “elder brother”; LSI aṇṇa uses the same form with an unrestricted “Brother” gloss.'),
    13: (['d3085'], 'DEDR 3085 gives Kannada tamma “younger brother”; the linked Badaga dictionary has tamma with that sense.'),
    14: (['d23'], 'DEDR 23 gives Kannada akka “elder sister”; the linked Badaga dictionary has the same form and kinship sense.'),
    15: (['d3015'], 'DEDR 3015(a) gives Kannada taṅge “younger sister” and Kodagu taŋge; LSI taŋe is a close survey rendering with a generic sister gloss.'),
    16: (['d399'], 'DEDR 399 gives Kannada āḷ “adult/person/man”; LSI ālu means “Man”. The linked Badaga ālu “milk” belongs to a different article and is excluded by sense.'),
    17: (['d4395'], 'DEDR 4395(a) gives Kannada peṇṇu and heṇṇ- forms for women; LSI heṇṇu has the woman sense.'),
    18: (['d4395'], 'DEDR 4395(a) gives Kannada peṇḍati/peṇḍiti “wife”; linked Badaga heṇdaru means “wife, woman”, close to LSI henḍaru.'),
    20: (['d1873'], 'DEDR 1873 gives Kannada kūsu “infant, babe”; linked Badaga kūsu “child, baby” supports LSI kusu, with vowel length unmarked in the survey.'),
    21: (['d4616'], 'DEDR 4616 gives Kannada maga “son”; linked Badaga maga has the same form and child/son meaning.'),
    22: (['d4395', 'd4616'], 'LSI hennu magalu “daughter” combines female hennu (DEDR 4395) with Kannada magaḷ “daughter” (DEDR 4616); both lexical components are preserved in order.'),
    23: (['d4395', 'd4616'], 'LSI heṇṇu makkalu “daughters” combines the female stem in DEDR 4395 and Kannada makkaḷu “children” in DEDR 4616; the plural morphology stays in the source form.'),
    24: (['d1173', 'd1711'], 'LSI ganḍu kudure “male horse” combines Kannada gaṇḍu “male” (DEDR 1173) and kudure “horse” (DEDR 1711). The spelling ganḍu is retained.'),
    25: (['d4395', 'd1711'], 'LSI heṇṇu kudure “mare” combines the female term in DEDR 4395 with Kannada kudure “horse” in DEDR 1711.'),
    26: (['d4593'], 'DEDR 4593 directly gives Kannada hōri “bull calf, bullock”; the linked Badaga dictionary has hōri “male calf, steer”, matching LSI “Bull”.'),
    31: (['d4395', 'd3650'], 'LSI heṇṇu nāi “bitch” combines the female term in DEDR 4395 with Kannada nāyi “dog” in DEDR 3650.'),
    32: (['d5152'], 'DEDR 5152 gives Kannada āḍu “goat”; LSI adu “Goats” has shortened/unspecified vowel and a plural elicitation gloss, so this is a qualified family-level link. The homophonous demonstrative adu is excluded by sense.'),
    33: (['d1017', 'd399'], 'LSI volle āl “good man” has a good stem comparable to Kannada oḷḷe (DEDR 1017), followed by āḷ “man” (DEDR 399). Initial v is retained as a survey qualification.'),
    34: (['d1017', 'd399'], 'LSI volle ālu glo “good men” preserves the same oḷḷe/āḷ lexical components as the singular (DEDR 1017, 399); glo is an unanalysed plural marker.'),
    36: (['d1017', 'd4395'], 'LSI volle heṇ gla “good women” has the oḷḷe “good” family (DEDR 1017) and heṇṇ- “female” family (DEDR 4395); the clipped noun and plural element are not normalized.'),
    37: (['d1942', 'd4259'], 'LSI keṭṭa huḍuga “bad boy” combines Kannada keṭṭa “bad” in DEDR 1942 and huḍuga “boy” in DEDR 4259.'),
    38: (['d1942', 'd4395'], 'LSI keṭṭa heṇṇu “bad girl” combines Kannada keṭṭa “bad” (DEDR 1942) and the female term heṇṇu (DEDR 4395).'),
    39: (['d621'], 'DEDR 621 gives Kannada udda “height/length”; linked Badaga udda is “high, long, tall”, matching the LSI “High” response.'),
    40: (['d5086'], 'DEDR 5086 gives Kannada mēle “above/up”; linked Badaga mēle has that same spatial sense.'),
    41: (['d1619'], 'DEDR 1619 gives Kannada keḷagu “lower side, below, down”; LSI kelage is a close Badaga survey form.'),
    45: (['d5020'], 'DEDR 5020(a) gives Kannada mun/munnu “before, in front”; linked Badaga munde has that meaning and matches LSI.'),
    46: (['d5020'], 'DEDR 5020(a) gives Kannada muñcu “precede, go before”; LSI munce is a close survey variant of “Before”.'),
    47: (['d4205'], 'DEDR 4205 gives Kannada hind- spatial/temporal “behind”; linked Badaga hinde directly matches the LSI form and sense.'),
    48: (['d5151'], 'DEDR 5151 gives Kannada yār “who”; linked Badaga yāru is the corresponding question word.'),
    49: (['d5151'], 'DEDR 5151 gives Kannada ēn “what”; LSI yēna adds the survey initial glide and final vowel.'),
    50: (['d5151'], 'DEDR 5151 gives Kannada yāke/ēke “why”; LSI yēka is the corresponding interrogative form.'),
}

holds = {
    19: 'LSI kūju “child” resembles the kūsu infant family (DEDR 1873) but the j/s difference is not established by a linked Badaga comparator; the possible related Koraga kūje is a different language.',
    27: 'LSI dana “cow” lacks a verified Badaga DEDR etymon or immediate donor; the linked Badaga search yielded no same-form cow analysis.',
    28: 'LSI hasu “cow” is homonymous with linked Badaga hasu “hunger” (DEDR 3825), which is semantically unrelated; cow ancestry remains unresolved.',
    29: 'Second LSI dana “cows” record retained separately; the same cow etymon/immediate donor gap remains.',
    30: 'Second LSI hasu “cows” record retained separately; the linked hunger homonym is excluded.',
    35: 'LSI colle heṇṇu “good woman” has a secure second component (DEDR 4395), but colle differs from the other volle “good” responses and from Kannada oḷḷe (DEDR 1017); hold the whole response rather than silently emend colle.',
    42: 'LSI klia “down” may be a clipped form of kelage (DEDR 1619), but the source spelling lacks the expected medial sequence; no direct comparator was found.',
    43: 'LSI vottura “near” has no secure DEDR family match or checked immediate donor.',
    44: 'LSI dura “far” resembles an Indo-Aryan dūra word, but no verified immediate donor node or borrowing path is established.',
}

assert not set(proposals) & set(holds)
assert set(proposals) | set(holds) == set(range(1, 51))
accepted = []
for i, (parents, evidence) in proposals.items():
    record = records[i - 1]
    row = dict(record=record, parent=parents[0], kind='component' if len(parents) > 1 else 'reflex',
               citation=record['Source'] + ''.join(f';dedr[{p[1:]}]' for p in parents),
               evidence=evidence, comparison='Full DEDR article, linked Badaga comparanda and survey source record checked.',
               status='pending-review')
    if len(parents) > 1:
        row['components'] = parents
    accepted.append(row)
held = [dict(record=records[i - 1], candidates=[], reason=reason) for i, reason in holds.items()]
(P / 'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review', accepted=accepted, held=held), ensure_ascii=False, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(held, ensure_ascii=False, indent=2) + '\n')
(P / 'unexamined-inventory.json').write_text(json.dumps(records[50:], ensure_ascii=False, indent=2) + '\n')
parents = {p for values, _ in proposals.values() for p in values}
counter = {'d3825', 'd1018', 'd1873', 'd5152'}
articles = {}
with gzip.open(P.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in parents | counter:
            articles[row['Entry_ID']] = row
assert parents <= articles.keys()
(P / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
print('first50', len(accepted), 'proposals', len(held), 'holds', len(records) - 50, 'unexamined')
