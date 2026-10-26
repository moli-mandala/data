"""Draft review-only Kurux LSI decisions; never writes accepted sidecars."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
R = json.loads((P / "survey-inventory.json").read_text())
assert len(R) == 148

articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as f:
    for a in csv.DictReader(f):
        articles[a["Entry_ID"]] = a

# Whole DEDR articles are family nodes. A survey spelling need not equal the
# reconstructed display head, but the relevant Kurux reflex must be in the prose.
anchors = {
    7: ("d5052", "DEDR 5052 has Kurux nubb ‘three persons’ beside mūnd ‘three things’; LSI nub is the person-counting variant, with final gemination omitted."),
    21: ("d399", "DEDR 399 directly gives Kurux āl ‘adult male, husband’ and ālas ‘adult male person’; the LSI bare āl has the matching man sense."),
    41: ("d4780", "DEDR 4780 gives Kurux māk ‘antelope, red deer’ and Malto máku ‘stag’; the LSI adds male sex to the same deer noun."),
    50: ("d4841", "DEDR 4841 gives Kurux meccā ‘high, tall, lofty’; LSI mecʰā is a compatible transcription, and the homepage’s other mecʰā ‘moustache’ belongs to an unrelated article."),
    51: ("d4841", "DEDR 4841 places Kurux mańyã̄ ‘above, overhead’ and Malto meṉye ‘up, above’ in this family; LSI mē̃yā is the spatial adverb, not a height noun."),
    52: ("d1619", "DEDR 1619 directly gives Kurux kiyyā ‘beneath, under’ and Brahui kī ‘below’; LSI kīyā has the spatial sense, unlike the homonymous ‘wooden box’ in DEDR 1559."),
    55: ("d2182", "DEDR 2182 directly gives Kurux xōxā ‘behind’ in both time and place and Malto qóq; the LSI form and meaning coincide."),
    56: ("d5151", "DEDR 5151 directly gives Kurux nē ‘who’ singular; the survey question preserves that interrogative."),
    57: ("d5151", "DEDR 5151 directly gives Kurux endrā ‘what thing’; the LSI question is this form."),
    58: ("d5151", "DEDR 5151 gives Kurux endr/endrā ‘what’ and the e- interrogative family; LSI endergē ‘why’ contains the interrogative plus a case element, not a separate inherited stem."),
    63: ("d4743", "DEDR 4743 gives Kurux malā/mal'ā ‘no, not’ and Malto mala; LSI mal ā reflects that negative response. The article’s distant Tamil/Telugu comparisons remain questioned."),
    64: ("d1419", "DEDR 1419 gives Kurux kānā ‘go’ with imperative kalā and past keras; the LSI elicitation kal agrees with the imperative stem."),
    65: ("d930", "DEDR 930 directly gives Kurux okknā ‘sit down’ and Malto oke; LSI okk is the elicited verb stem."),
    69: ("d5160", "DEDR 5160 gives Kurux ēn ‘I’ with oblique eŋg-; the LSI nominative is exact."),
    70: ("d5160", "DEDR 5160 gives Kurux ēn ‘I’ with oblique eŋg-; LSI eŋhai ‘my’ is the possessive based on that oblique stem, not an independent root."),
    71: ("d5154", "DEDR 5154 directly gives Kurux ēm ‘we’ exclusive, with oblique em-; preserve the exclusive sense rather than conflating inclusive nām."),
    72: ("d5154", "DEDR 5154 gives Kurux ēm ‘we’ exclusive and oblique em-; LSI emhai ‘our’ continues the oblique plus possessive marking."),
    73: ("d3684", "DEDR 3684 gives Kurux nīn ‘thou’ with oblique niŋg-; LSI niŋhai ‘thy’ is the matching possessive form."),
    74: ("d3688", "DEDR 3688 directly gives Kurux nīm ‘you’ plural/honorific with oblique nim-; this is distinct from singular nīn in 3684."),
    75: ("d5270", "DEDR 5270 gives Kurux barnā ‘come’ and Malto bare, with wider Dravidian vā- comparanda; LSI bar is the verb stem."),
    76: ("d3675", "DEDR 3675 gives Kurux ijnā (ijjas) ‘stand, stop, halt’; LSI ijj preserves the cited present stem."),
    77: ("d2426", "DEDR 2426 gives Kurux khē'enā (keccas) ‘die’ and Malto keye; LSI kʰē is the corresponding stem."),
    78: ("d2598", "DEDR 2598 gives Kurux ci'inā (ciccas) ‘give, allow’ and Malto ciye; LSI ci is the elicited root."),
    79: ("d4473", "DEDR 4473 gives Kurux boŋgnā ‘run’ and Malto boṉg̣e ‘run, flee’; LSI boŋg is the same verb stem."),
    123: ("d3688", "DEDR 3688 gives Kurux nīm ‘you’ with oblique nim-; LSI nimhai ‘your’ is a possessive on that plural/honorific base."),
    124: ("d1", "DEDR 1 gives Kurux ās ‘that man, he’; the LSI pronoun matches directly."),
    125: ("d1", "DEDR 1 gives Kurux ās ‘he’ and the a- deictic base; LSI ās ghai ‘his’ adds a possessive marker."),
    126: ("d1", "DEDR 1 gives Kurux ār ‘those persons, they’; the LSI pronoun is exact."),
    127: ("d1", "DEDR 1 gives Kurux ār ‘those persons, they’; LSI ār gahi ‘their’ adds a possessive marker."),
    128: ("d2023", "DEDR 2023 directly gives Kurux xekkhā ‘hand, arm’ and Malto qeqe; LSI xekkhā is exact."),
    129: ("d1943", "DEDR 1943 gives Kurux xeḍḍ ‘foot, leg’ and Malto qeḍu; LSI xed abbreviates the retroflex/geminate notation while keeping the foot sense."),
    130: ("d4570", "DEDR 4570 directly gives Kurux pannā ‘iron’; although most of the article concerns ‘gold/metal’, the Kurux iron reflex is explicit."),
    138: ("d3645", "DEDR 3645 directly gives Kurux nād ‘devil, evil spirit; idol’ and Malto náde deity-stone; LSI nāḍ differs only in retroflex notation."),
    144: ("d5490", "DEDR 5490 directly gives Kurux berxā ‘cat’ and Malto berge, beside broader Dravidian cat terms; LSI berxā is exact."),
    148: ("d1514", "DEDR 1514 directly gives Kurux cicc ‘fire’ and Malto cicu; LSI cic is the same form with simplified final gemination."),
}

pronouns = {111: "d5160", 112: "d3684", 113: "d1", 114: "d5154", 115: "d3688", 116: "d1",
            117: "d5160", 118: "d3684", 119: "d1", 120: "d5154", 121: "d3688", 122: "d1"}
pronoun_labels = {"d5160": "ēn ‘I’", "d3684": "nīn ‘thou’", "d1": "ās/ār ‘he/they’", "d5154": "ēm ‘we’ exclusive", "d3688": "nīm ‘you’"}

accepted = []
for i, (parent, evidence) in anchors.items():
    record = R[i - 1]
    accepted.append({"record": record, "parent": parent, "kind": "reflex",
                     "citation": record["Source"] + f";dedr[{parent[1:]}]",
                     "evidence": evidence, "comparison": "Full DEDR article and homepage exact-form search checked.",
                     "status": "pending-review"})
for i, pronoun in pronouns.items():
    record = R[i - 1]
    tense = "present" if i <= 116 else "past"
    evidence = (f"LSI {record['Form']} is a subject–predicate clause: {pronoun_labels[pronoun]} "
                f"plus the {tense} go stem of Kurux kānā. DEDR 1419 explicitly includes past keras "
                "alongside present/future forms; person/tense endings are ordinary inflection.")
    accepted.append({"record": record, "parent": pronoun, "kind": "component",
                     "components": [pronoun, "d1419"],
                     "citation": record["Source"] + f";dedr[{pronoun[1:]}];dedr[1419]",
                     "evidence": evidence, "evidence_parent": "d1419",
                     "comparison": "Ordered subject and predicate verified in full DEDR articles.",
                     "status": "pending-review"})

loan_groups = {
    "Indo-Aryan-looking numerals; immediate donor unverified": [1,2,3,4,5,6,67],
    "Kinship form likely affected by contact; no verified immediate donor node": [8,9,10,11,12,13,14,15,16,17,18,19,20,22],
    "Kinship form does not securely match a DEDR family or its exact morphological analysis": [23,24,25],
    "Contact-looking animal name or metal term; immediate donor not established": [26,27,28,29,30,31,32,33,131,132,140,141,142,143,147],
    "DEDR 2916 marks Kurux allā ‘dog’ with a question mark; whole response/component analysis uncertain": [34,35,36,37],
    "Multi-part animal response includes an unetymologised sex or plural element": [38,39,40,42,43],
    "Multi-part good/bad response includes an unverified qualifying root": [44,45,46,47,48,49],
    "No secure matching family in exact-form search and complete DEDR prose": [53,54,59,60,61,62,66,68,92,133,135,136,137,139,145,146],
    "Full copular clause has no verified etymon for the predicate; partial pronoun link deferred": list(range(80,92)),
    "Full beat clause has no verified etymon for lau ‘beat’; partial pronoun link deferred": list(range(93,111)),
    "Questioned DEDR 2813 relates Kurux jō̃xas ‘lad, youth; servant’; LSI ‘slave’ requires review": [134],
}
reasons = {}
for reason, indices in loan_groups.items():
    for i in indices:
        assert i not in reasons, i
        reasons[i] = reason
held = []
covered = {R.index(x["record"]) + 1 for x in accepted}
for i, record in enumerate(R, 1):
    if i in covered:
        continue
    assert i in reasons, (i, record["Form"])
    held.append({"record": record, "candidates": [], "reason": reasons[i]})

assert len(covered) + len(held) == len(R)
assert len({x["record"]["ID"] for x in accepted + held}) == len(R)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
ids = {x["parent"] for x in accepted} | {y for x in accepted for y in x.get("components", [])}
(P / "primary-evidence.json").write_text(json.dumps({i: articles[i] for i in sorted(ids)}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"accepted_records": len(accepted), "assignment_rows": sum(len(x.get("components", [x["parent"]])) for x in accepted), "holds": len(held), "article_count": len(ids)}))
