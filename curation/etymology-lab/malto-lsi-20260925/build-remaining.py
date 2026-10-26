"""Extend the validated first-fifty checkpoint to the full LSI Malto residual."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
survey = json.loads((P / 'survey-inventory.json').read_text())
first = json.loads((P / 'checkpoint-001-decisions.json').read_text())
articles = json.loads((P / 'primary-evidence.json').read_text())
assert len(survey) == 165
assert len(first['accepted']) == 9 and len(first['held']) == 41
assert {x['record']['ID'] for x in first['accepted'] + first['held']} == {r['ID'] for r in survey[:50]}

direct = {}

def add(indices, parent, evidence):
    if isinstance(indices, int):
        indices = [indices]
    assert parent in articles
    for i in indices:
        assert i not in direct
        direct[i] = (parent, evidence)

add(53, 'd4841', 'DEDR 4841 directly gives Malto mecge “height” and mece “above, up”. LSI mecge “high” uses the same upper/height stem adjectivally.')
add(54, 'd4841', 'DEDR 4841 directly gives Malto mece “above, up”, exactly the LSI directional form.')
add(55, 'd4205', 'DEDR 4205 directly gives Malto pisti “the lower” and pisi “below”; LSI pisti “down” uses that directional family.')
add([58, 144], 'd2182', 'DEDR 2182 directly gives Malto qóq “behind, at the back” and qóqe “the back”. LSI qōq/qoqeθ match the directional and noun stems, with length or nominal-ending differences in the survey spelling.')
add(59, 'd5151', 'DEDR 5151 directly lists Malto nére(h) “who, which man”; LSI nēreh is the same interrogative stem.')
add(60, 'd5151', 'DEDR 5151 directly lists Malto indru “which thing/what”; LSI indru matches the cited form.')
add(61, 'd5151', 'DEDR 5151 directly lists Malto indrik “why”; LSI indrik matches the cited form.')
add(65, 'd4743', 'DEDR 4743 directly gives Malto mala “no, not” and negative mal-; LSI mala is an exact negative match. Cross-language comparisons in the article are questioned, but the Malto form itself is explicit.')
add(66, 'd1419', 'DEDR 1419 gives Malto kále “to go, come to”; LSI kāl is the shorter go stem, with a survey vowel-length difference.')
add(68, 'd930', 'DEDR 930 directly gives Malto oke “to sit, be situated”; LSI ok is the elicited short stem.')
add(72, 'd5160', 'DEDR 5160 directly gives Malto én “I”; LSI ēn differs only in recorded vowel quantity.')
add(73, 'd5160', 'DEDR 5160 directly gives Malto én with oblique eṉg-; LSI eŋ “my” uses that oblique first-person stem.')
add(74, 'd3647', 'DEDR 3647 directly gives Malto nám (oblique nam-) as inclusive “we”. LSI nām lacks a clusivity gloss, so the article-specific inclusive value is retained in the evidence.')
add(75, 'd5154', 'DEDR 5154 directly gives Malto ém (oblique em-) as exclusive “we”. LSI ēm lacks a clusivity gloss, so the article-specific exclusive value is retained in the evidence.')
add(76, 'd3647', 'DEDR 3647 gives Malto nám with oblique nam- for inclusive “we”; LSI nam “our” is the expected possessive use of that oblique stem.')
add(77, 'd5154', 'DEDR 5154 gives Malto ém with oblique em- for exclusive “we”; LSI em “our” is the expected possessive use of that oblique stem.')
add(78, 'd3684', 'DEDR 3684 gives Malto nín “thou” with oblique niṉg-; LSI niŋ “thy” uses the oblique second-singular stem.')
add([79, 128], 'd3688', 'DEDR 3688 directly gives Malto ním “you (plural)” with oblique nim-. LSI nīm “you” and nim “your” follow that direct/oblique paradigm; the source gloss does not specify number.')
add(80, 'd5270', 'DEDR 5270 directly gives Malto bare “to come” within the var-/bar- family; LSI bar is the short come stem.')
add(81, 'd3675', 'DEDR 3675 directly gives Malto ile (ij-) “to stand”; LSI il is the short stand stem.')
add(82, 'd2426', 'DEDR 2426 directly gives Malto keye (kec-) “to die”; LSI ke is a short elicited stem within that same paradigm.')
add(83, 'd2598', 'DEDR 2598 directly gives Malto ciye (cic-) “to give away”; LSI ci is the short give stem.')
add(129, 'd1', 'DEDR 1 directly gives Malto áh “that man/he”; LSI āh “he” is the same distal third-person stem.')
add(130, 'd1', 'DEDR 1 gives Malto áh “that man/he”; LSI ahi “his” is its possessive/case form, not a separate lexical derivation.')
add(131, 'd1', 'DEDR 1 directly gives Malto ár “those persons/they”; LSI ār matches the plural third-person form.')
add(132, 'd1', 'DEDR 1 gives Malto ár “those persons/they”; LSI āri “their” is its possessive/case form, not a separate lexical derivation.')
add(134, 'd1943', 'DEDR 1943 directly gives Malto qeḍu “the legs, the feet”; LSI qeḍu “foot” selects a singular questionnaire sense from that body-part family.')
add(135, 'd4909', 'DEDR 4909 gives Malto muso “snot; the nose”; LSI musoθ “nose” retains the same stem with final nominal -θ.')
add(136, 'd1159', 'DEDR 1159 directly gives Malto qanu “eye”; LSI qanuθ adds final nominal -θ.')
add(137, 'd3296', 'DEDR 3296 directly gives Malto toro “mouth”; LSI toroθ adds final nominal -θ.')
add(138, 'd3986', 'DEDR 3986 directly gives Malto palu “the teeth”; LSI paluθ “tooth” retains the stem with a nominal ending, while singular versus plural differs across elicitation glosses.')
add(140, 'd3103', 'DEDR 3103 directly gives Malto tali “hair of head”; LSI taliθ “hair” adds final nominal -θ.')
add(141, 'd1630', 'DEDR 1630 gives Malto kuku “head”; LSI qukuθ has survey q for dictionary k and final nominal -θ.')
add(142, 'd3064', 'DEDR 3064 directly gives Malto taṛte “tongue”; LSI taṛteθ adds final nominal -θ.')
add(143, 'd4494', 'DEDR 4494 directly gives Malto puṛa “belly, bowels”; LSI puṛaθ adds final nominal -θ. The article notes a Pkt. comparison without settling contact.')
add(151, 'd187', 'DEDR 187 directly gives Malto amu “water”; LSI amuθ adds final nominal -θ. The article questions a Sanskrit ambu comparison, which this link does not decide.')
add(156, 'd334', 'DEDR 334 directly gives Malto óyu “cow, ox”; this second LSI ōyuθ record is a distinct source entry for the same cow form reviewed in checkpoint 001.')
add(158, 'd5490', 'DEDR 5490 directly gives Malto berge “cat”; LSI bergeθ adds final nominal -θ.')
add(162, 'd5517', 'DEDR 5517 directly gives Malto béru “sun” within the heat family; LSI bēruθ adds final nominal -θ.')
add(163, 'd5496', 'DEDR 5496 directly gives Malto bilpu “moon” within the bright/white family; LSI bilpuθ adds final nominal -θ.')
add(164, 'd4876', 'DEDR 4876 directly gives Malto bínḍke “star”; LSI bīṇḍkeθ has a vowel-length and nasal-place spelling difference plus final nominal -θ.')
add(165, 'd1514', 'DEDR 1514 directly gives Malto cicu “fire” beside Kurux cicc; LSI cicuθ adds final nominal -θ. The article also compares a Prakrit fire word, leaving contact unresolved.')

components = {}
pronouns = {116: 'd5160', 117: 'd3684', 118: 'd1', 119: 'd5154', 120: 'd3688', 121: 'd1',
            122: 'd5160', 123: 'd3684', 124: 'd1', 125: 'd5154', 126: 'd3688', 127: 'd1'}
for i, pronoun in pronouns.items():
    record = survey[i - 1]
    form = record['Form']
    pron, verb = form.split(' ', 1)
    assert pronoun in articles and 'd871' in articles
    components[i] = ([pronoun, 'd871'],
        f'DEDR {pronoun[1:]} contains the Malto {pron} pronoun or its case paradigm; DEDR 871 directly gives Malto éke “to go, move”. LSI {verb} is a finite go form following the pronoun. The two source words remain ordered components, while person and tense endings are inflectional.')

holds = {
    51: 'bāna maqe “bad boy” contains the maqe child stem (DEDR 4616), but bāna “bad” lacks a verified DEDR family or immediate donor; the whole phrase remains unassigned.',
    52: 'bāna maqiθ “bad girl” contains the maqi girl stem (DEDR 4616), but bāna “bad” lacks a verified DEDR family or immediate donor.',
    56: 'bahano “near” lacks a matching full DEDR article; DEDR 79 Malto aṭge “near” is a different stem.',
    57: 'agdu “before” lacks a verified full DEDR family or immediate donor.',
    62: 'ado “and” has no verified same-stem DEDR article or immediate donor.',
    63: 'je “but” is too short and lacks a diagnostic same-meaning DEDR match or immediate donor.',
    64: 'ō̃ō̃ “yes” is an expressive response; no sound-supported DEDR family or immediate donor was established.',
    67: 'lap “eat” does not match the Malto eat verbs in the searched DEDR articles; an inherited parent or immediate donor remains unknown.',
    69: 'kōṛy ond “twenty” is a two-part numeral phrase; neither the score word nor ond has a complete source-order etymology.',
    70: 'aṛai kōṛi “fifty” is a two-part numeral phrase; aṛai and kōṛi require independent comparison before linking.',
    71: 'pac kōṛi “hundred” combines the held pac “five” with an unresolved score word, so the full numeral cannot be assigned.',
    84: 'boŋg “run” has no verified matching Malto DEDR reflex or immediate donor.',
    133: 'tetuθ “hand” does not match Malto qeqe “hand” in DEDR 2023; an independent root or donor is unresolved.',
    139: 'qeθvuθ “ear” has no verified Malto DEDR counterpart with its full consonant sequence; possible nominal -θ does not resolve the stem.',
    145: 'lohaθ “iron” resembles a widespread Indo-Aryan metal term, but the immediate donor into this Malto survey variety is not established.',
    146: 'sonaθ “gold” resembles a widespread Indo-Aryan metal term, but the immediate donor into this Malto survey variety is not established.',
    147: 'candiθ “silver” resembles a widespread Indo-Aryan metal term, but the immediate donor into this Malto survey variety is not established.',
    148: 'golām “slave” resembles a widespread Indo-Aryan/Persian contact term, but the immediate donor into Malto is not established.',
    149: 'gosayiθ “God” resembles a regional religious title, but the source and immediate donor into Malto are unresolved.',
    150: 'buteθ “devil” resembles a regional spirit term, but a specific immediate donor or inherited family is not verified.',
    152: 'aḍaθ “house” has no direct Malto whole-article DEDR match or verified immediate donor.',
    153: 'goroθ “horse” repeats the held goṛoθ horse noun with an alternate survey spelling; no immediate donor has been verified.',
    154: 'gada goṛoθ “ass” combines an unresolved modifier with the held horse term; neither the compound nor its donor chain is established.',
    155: 'uṭeθ “camel” resembles regional Indo-Aryan camel terms, but the immediate donor into Malto has not been verified.',
    157: 'aleθ “dog” repeats the earlier source form; DEDR 2916 explicitly questions the Malto ale comparison, so this duplicate record remains held.',
    159: 'pujuθ “bird” has no verified Malto DEDR match or immediate donor.',
    160: 'qēr laŋgaθ “cock” is a two-word animal term with neither component securely matched to a full DEDR article or immediate donor.',
    161: 'batakeθ “duck” resembles a regional Indo-Aryan bird term, but the immediate donor into this Malto variety is not established.',
}

for i in range(85, 97):
    r = survey[i - 1]
    holds[i] = (f'{r["Form"]} “{r["Gloss"]}” is a two-word pronoun + finite copula response. The pronoun has a DEDR family, '
                'but the ḍōk-/ḍokk- “be” stem has no verified full article; the complete expression is held rather than flattened into a pronoun edge.')
for i in range(97, 116):
    r = survey[i - 1]
    holds[i] = (f'{r["Form"]} “{r["Gloss"]}” belongs to the LSI beat paradigm, but baj- has no verified Malto DEDR family or immediate donor. '
                + ('The finite response also includes a pronoun; no full source-order component analysis is ready.' if ' ' in r['Form'] else 'The verb itself remains unassigned.'))

assert set(direct).isdisjoint(components)
assert set(direct).isdisjoint(holds)
assert set(components).isdisjoint(holds)
assert set(direct) | set(components) | set(holds) == set(range(51, 166))

accepted = list(first['accepted'])
held = list(first['held'])
for i in range(51, 166):
    record = survey[i - 1]
    if i in direct:
        parent, evidence = direct[i]
        accepted.append(dict(record=record, parent=parent, kind='reflex', citation=record['Source'] + ';dedr[' + parent[1:] + ']', evidence=evidence, status='pending-review'))
    elif i in components:
        parents, evidence = components[i]
        accepted.append(dict(record=record, parent=parents[0], components=parents, kind='component', citation=record['Source'] + ''.join(';dedr[' + p[1:] + ']' for p in parents), evidence=evidence, status='pending-review'))
    else:
        held.append(dict(record=record, candidates=[], reason=holds[i]))

assert {x['record']['ID'] for x in accepted + held} == {r['ID'] for r in survey}
assert all(p in articles for x in accepted for p in x.get('components', [x['parent']]))
assert accepted[:len(first['accepted'])] == first['accepted']
assert held[:len(first['held'])] == first['held']
(P / 'inventory.json').write_text(json.dumps(survey, ensure_ascii=False, indent=2) + '\n')
(P / 'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review', accepted=accepted, held=held), ensure_ascii=False, indent=2) + '\n')
print({'proposedRecords': len(accepted), 'assignmentRows': sum(len(x.get('components', [x['parent']])) for x in accepted), 'held': len(held)})
