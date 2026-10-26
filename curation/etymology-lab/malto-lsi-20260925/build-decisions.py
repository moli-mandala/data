"""Prepare the first fifty LSI Malto records for review without saving links."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'inventory.json').read_text())
assert len(records) == 50

direct = {
    15: ('d474', 'DEDR 474 explicitly gives Malto iwr “two persons”, exactly the LSI form ivr. The LSI questionnaire gloss “Two Fathers” is contextual; the response itself supplies only the personal numeral, so no father component is inferred.'),
    22: ('d4756', 'DEDR 4756 directly gives Malto muṛse “man, male”, matching the LSI spelling and meaning.'),
    23: ('d4198', 'DEDR 4198 directly gives Malto peli “woman”; LSI peliθ adds a final nominal ending while retaining the same stem and gloss.'),
    25: ('d4616', 'DEDR 4616 gives Malto maqe “boy” and maqo “young, small” in the wider child family. LSI maqeh “child” is a broad use of the directly recorded maqe stem.'),
    27: ('d4616', 'DEDR 4616 is a whole child/offspring article: it gives Malto maqo “young, small” and maqi “girl”, and Tamil makaḷ “daughter”. LSI maqoθ “daughter” has the maqo stem with nominal -θ; the article-level link does not claim DEDR directly glosses Malto maqo as daughter.'),
    34: ('d334', 'DEDR 334 directly gives Malto óyu “cow, ox”. LSI ōyuθ matches the cow sense with final nominal -θ.'),
    46: ('d2294', 'DEDR 2294 directly gives Malto canju “deer” alongside Kurux canju “antelope”; LSI cañjuθ preserves the stem with palatal nasal and nominal -θ. The article compares Sanskrit cañcu, so borrowing versus inheritance remains unresolved.'),
}
component = {
    49: (['d848', 'd4198'], 'DEDR 848 directly gives Malto éṛu “good”, and DEDR 4198 gives Malto peli “woman”. LSI ēṛu peliθ combines these source-order components, with final nominal -θ.'),
    50: (['d848', 'd4198'], 'DEDR 848 directly gives Malto éṛu “good”, and DEDR 4198 gives Malto peli “woman”. LSI ēṛu peler has the same adjective followed by the plural woman stem; plurality is inflectional, not a new etymon.'),
}
holds = {
    1: 'pac “five” resembles regional Indo-Aryan five forms, but an immediate donor into the survey variety has not been identified.',
    2: 'cō “six” has no confirmed whole-article DEDR match or securely identified immediate donor.',
    3: 'sāt “seven” resembles regional Indo-Aryan numeral forms, but its immediate donor is not established.',
    4: 'aṭ “eight” resembles regional Indo-Aryan numeral forms; the exact source and transmission path need verification.',
    5: 'nō “nine” resembles regional Indo-Aryan numeral forms; the immediate donor remains unidentified.',
    6: 'ēṇḍ ond “one” has two written elements. DEDR 990 contains Dravidian one forms, but ēṇḍ has not been parsed and the whole response cannot be assigned to one stem.',
    7: 'das “ten” is widespread in regional Indo-Aryan languages; the immediate donor into this Malto variety has not been established.',
    8: 'ēṇḍ is “two” includes Malto -is “two things” in DEDR 474, but the preceding ēṇḍ is unparsed. A whole-response edge would flatten a two-element expression.',
    9: 'tīn “three” resembles regional Indo-Aryan numeral forms, but the immediate donor is not secure.',
    10: 'cār “four” resembles regional Indo-Aryan numeral forms, but the immediate donor is not secure.',
    11: 'abba “father” is not directly attested for Malto in a verified DEDR article; a possible contact or kinship family has not been resolved.',
    12: 'abba ki “of a father” combines an unverified father stem and genitive particle; the whole expression is not one inherited stem.',
    13: 'abba bahak “to a father” combines the unresolved father stem and allative material requiring separate analysis.',
    14: 'abba bahante “from a father” combines the unresolved father stem and ablative material requiring separate analysis.',
    16: 'abba saber “fathers” contains unresolved abba plus a separate plural/person marker saber; neither has a verified full analysis.',
    17: 'abba saber ki “of fathers” has father, plural/person and genitive elements; no complete source-order component analysis is ready.',
    18: 'abba saber bahak “to fathers” has father, plural/person and allative elements; no complete source-order component analysis is ready.',
    19: 'abba saber inte “from fathers” has father, plural/person and ablative elements; no complete source-order component analysis is ready.',
    20: 'baya “brother” has no verified DEDR family or supported immediate donor; resemblance to regional Indo-Aryan kin terms alone is insufficient.',
    21: 'bāyiθ “sister” resembles the brother term but its stem and final nominal material have no verified family or donor.',
    24: 'ḍāniθ “wife” has no verified DEDR family or immediate donor for this Malto form.',
    26: 'deh “son” does not match Malto qade “son” in DEDR 1411; a separate etymon or donor remains unverified.',
    28: 'eŋgad bagter “daughters” is a two-part response with no verified daughter-stem correspondence or complete component analysis.',
    29: 'goṛoθ “horse” resembles regional Indo-Aryan horse terms, but its immediate donor into Malto has not been verified.',
    30: 'ḍaḍi goṛoθ “mare” adds a female modifier to the unresolved horse noun; no complete component analysis is ready.',
    31: 'goṛo gahṇḍiθ “horses” adds separate plural material to the unresolved horse noun; no whole-stem etymology is justified.',
    32: 'ḍaḍi goṛo gahṇḍiθ “mares” combines female modifier, horse noun and plural expression with unresolved donor/components.',
    33: 'saṇḍeθ “bull” resembles regional Indo-Aryan bull terms, but DEDR 1173 Malto geṇḍa means “male” and does not establish this exact noun; the immediate donor is unknown.',
    35: 'saṇḍe gahṇḍiθ “bulls” combines the unresolved bull noun and an independent plural expression.',
    36: 'ōyu gahṇḍiθ “cows” contains óyu “cow” in DEDR 334, but the separate plural expression gahṇḍiθ lacks a verified component family.',
    37: 'DEDR 2916 prints “? Malt. ale” for “dog” and also questions comparison with DEDR 3650. The Malto article placement is explicitly uncertain, so the LSI aleθ is held.',
    38: 'ḍaḍi aleθ “bitch” adds a female modifier to ale, whose placement in DEDR 2916 is expressly questioned; the modifier is also unresolved.',
    39: 'al gahṇḍiθ “dogs” combines the questioned Malto dog stem (DEDR 2916) with an unresolved plural expression.',
    40: 'al ḍaḍi gahṇḍiθ “bitches” combines questioned dog stem, female modifier and plural expression; no complete component analysis is ready.',
    41: 'bokṛa ēṛeθ “he goat” includes Malto éṛe “she-goat” in DEDR 5152, but bokṛa is an unresolved male modifier or donor; the two-part response is held.',
    42: 'ēṛ ḍaḍiθ “female goat” contains the DEDR 5152 goat stem, but the separate female term ḍaḍiθ has not been securely sourced.',
    43: 'ēṛ gahṇḍiθ “goats” contains the DEDR 5152 goat stem, but the separate plural expression gahṇḍiθ has not been securely sourced.',
    44: 'bokṛā cañjuθ “male deer” contains Malto canju “deer” (DEDR 2294) but bokṛā is an unresolved male modifier or contact term.',
    45: 'ḍaḍi cañjuθ “female deer” contains Malto canju “deer” (DEDR 2294) but the separate female modifier is unresolved.',
    47: 'ēṛu māle “good man” includes Malto éṛu “good” (DEDR 848), but māle is not the Malto muṛse “man” recorded by DEDR 4756. The noun is unresolved.',
    48: 'ēṛu maler “good men” includes Malto éṛu “good” (DEDR 848), but maler is not the Malto muṛse “man” recorded by DEDR 4756. The plural noun is unresolved.',
}
assert set(direct).isdisjoint(component)
assert set(direct).isdisjoint(holds)
assert set(component).isdisjoint(holds)
assert set(direct) | set(component) | set(holds) == set(range(1, 51))

accepted = []
held = []
for i, record in enumerate(records, 1):
    if i in direct:
        parent, evidence = direct[i]
        accepted.append(dict(record=record, parent=parent, kind='reflex', citation=record['Source'] + ';dedr[' + parent[1:] + ']', evidence=evidence, status='pending-review'))
    elif i in component:
        parents, evidence = component[i]
        accepted.append(dict(record=record, parent=parents[0], components=parents, kind='component', citation=record['Source'] + ''.join(';dedr[' + p[1:] + ']' for p in parents), evidence=evidence, status='pending-review'))
    else:
        held.append(dict(record=record, candidates=[], reason=holds[i]))

articles = json.loads((P / 'primary-evidence.json').read_text())
assert all(p in articles for x in accepted for p in x.get('components', [x['parent']]))
(P / 'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review', accepted=accepted, held=held), ensure_ascii=False, indent=2) + '\n')
print({'proposedRecords': len(accepted), 'assignmentRows': sum(len(x.get('components', [x['parent']])) for x in accepted), 'held': len(held)})
