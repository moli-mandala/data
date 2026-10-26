"""Prepare the remaining Telugu LSI records for scholarly review, without saving links."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
rows = json.loads((P / "inventory.json").read_text())
assert len(rows) == 133

# Keys are one-based indices in the original residual inventory. A dictionary
# article is the parent family; the note identifies the actually printed reflex.
parents = {
    51: ("d2559", "DEDR 2559 explicitly gives Te. lēdu ‘no, none’ in the negative-existence paradigm; LSI lēdŭ is that form."),
    52: ("d4572", "DEDR 4572 gives Te. pōvu with stems pō- and pōy- ‘go’; LSI pō is the cited short stem."),
    54: ("d474", "DEDR 474 explicitly gives Te. iruvadi, colloquial iruvai ‘twenty’; LSI iruvai matches the colloquial form."),
    55: ("d2826", "DEDR 2826 gives Te. yābhai ‘fifty’; LSI yābʰai represents the same form with aspirated survey spelling."),
    57: ("d3647", "DEDR 3647 gives Te. manamu ‘we’ with oblique mana-; LSI manā ‘our’ is an oblique possessive form of that inclusive plural pronoun."),
    58: ("d5154", "DEDR 5154 gives Te. mēmu ‘we’ with oblique man-/mā-; LSI mā ‘our’ matches the possessive stem mā-."),
    59: ("d3684", "DEDR 3684 explicitly gives Te. nīvu ‘thou’ with oblique nī-/nin-; LSI nīvu matches the nominative form."),
    60: ("d3684", "DEDR 3684 gives Te. nīvu with oblique nī-/nin-; LSI nī ‘thy’ is the possessive stem nī-."),
    61: ("d3688", "DEDR 3688 gives Te. mīru ‘you’ with oblique mī-/mim-; LSI mīru matches the cited form."),
    63: ("d3675", "DEDR 3675 gives Te. nilucu ‘stand’ in the nil- family; LSI nilʦ is a reduced elicited stem with the same sense."),
    64: ("d2426", "DEDR 2426 gives Te. caccu with stem cā- ‘die’; LSI ʦā is that short stem in survey transcription."),
    65: ("d2598", "DEDR 2598 gives Te. iccu with stems ī̆-/iyy-/ivv- ‘give’; LSI iy is the short iyy- stem."),
    66: ("d713", "DEDR 713 gives Te. uṟuku ‘jump, run away’; LSI uruk ‘run’ matches the stem, though the survey gloss is broader."),
    79: ("d2063", "DEDR 2063 gives Te. koṭṭu ‘beat, strike, knock’; LSI koṭṭ is the elicited verbal stem."),
    104: ("d3688", "DEDR 3688 gives Te. mīru with oblique mī-/mim-; LSI mī ‘your’ matches the possessive stem mī-."),
    105: ("d1", "The full demonstrative article DEDR 1 explicitly gives Te. vā̃ḍu ‘that man, he’; LSI vāḍu matches the pronoun."),
    106: ("d1", "DEDR 1 explicitly gives Te. vā̃ḍu with oblique vāni-; LSI vāni ‘his’ matches that stem."),
    107: ("d1", "DEDR 1 explicitly gives Te. vāru ‘those persons, they’; LSI vāru matches the pronoun."),
    108: ("d1", "DEDR 1 explicitly gives Te. vāru with oblique vāri-; LSI vāri ‘their’ matches that stem."),
    112: ("d3633", "DEDR 3633 gives Te. nālika/nāluka/nālka ‘tongue’; LSI nālukĕ is a survey variant of nāluka."),
    115: ("d486", "DEDR 486 explicitly gives Te. inumu ‘iron’; LSI inumŭ matches it."),
    117: ("d5496", "The full DEDR 5496 white/shine article gives Te. veṇḍi ‘silver’; LSI veṇḍĭ matches that reflex."),
    122: ("d494", "DEDR 494 explicitly gives Te. illu ‘house, dwelling’; LSI illŭ matches it."),
    123: ("d1711", "DEDR 1711 gives Te. guṟṟamu ‘horse’; LSI gurramŭ matches. The article raises a possible Indo-Aryan origin, which this family assignment does not settle."),
    124: ("d1364", "DEDR 1364 explicitly gives Te. gāḍida ‘ass’; LSI gāḍidĕ matches it. The article compares Sanskrit gardabha but does not establish the immediate donor."),
    126: ("d4180", "DEDR 4180 explicitly gives Te. pilli ‘cat’; LSI pillĭ matches it. The article also compares CDIAL *billa-, so ultimate origin remains open."),
    128: ("d4373", "DEDR 4373 gives Te. punju ‘cock’; LSI punʣŭ matches its survey affricate spelling."),
    132: ("d2646", "DEDR 2646 explicitly gives Te. cukka ‘star’; LSI ʦukkă matches the survey affricate spelling."),
}

specific_holds = {
    53: (["d3675"], "LSI kūrʦ ‘sit’ has no direct form match in the searched DEDR articles; the stand family is semantically different."),
    56: (["d3647", "d5154"], "LSI nā ‘my’ needs a first-singular pronoun article; the nearby DEDR 3647/5154 articles are plural and do not justify this assignment."),
    62: (["d5270"], "DEDR 5270 has Tamil imperative vā ‘come’ but gives Telugu vaccu (rā-); LSI Telugu vā is not established as the cited Telugu reflex."),
    109: (["d2023"], "DEDR 2023 gives Te. ceyyi ‘hand’, but LSI eheyyi has an unexplained initial e- and missing c; do not normalize it silently."),
    110: ([], "LSI pādamu ‘foot’ is likely an Indo-Aryan loan, but an immediate donor and graph parent were not verified."),
    111: ([], "LSI veṇṭruka ‘hair’ lacks a matching full DEDR article or verified immediate donor."),
    113: (["d1198"], "The similar DEDR 1198 Te. kadupu means ‘mass, multitude’, not the LSI kaḍupŭ ‘belly’; the homograph is not enough."),
    114: (["d5488"], "DEDR 5488 gives Te. ven(n)u ‘back’ and Kannada bennu, but LSI Telugu bennŭ matches the Kannada onset; the source transcription or local variant needs confirmation."),
    116: ([], "LSI baŋgāramŭ ‘gold’ has no verified immediate donor or full DEDR family."),
    118: ([], "LSI naukarŭ ‘slave’ appears to be a loan; an immediate donor for this survey form was not verified."),
    119: ([], "LSI gollavāḍŭ ‘shepherd’ appears compound or caste-based; both components and its history need verification."),
    120: ([], "LSI dēvuḍŭ ‘god’ appears to involve Sanskrit deva, but a direct immediate donor and inherited-versus-loan analysis were not established."),
    121: ([], "LSI dayyamŭ ‘devil’ lacks a verified full DEDR family or immediate donor."),
    125: ([], "LSI loṭṭipiṭṭă ‘camel’ is opaque and lacks a verified full DEDR family or immediate donor."),
    127: ([], "LSI pakṣĭ ‘bird’ likely reflects an Indo-Aryan lexeme; its immediate donor and graph parent were not established."),
    129: ([], "LSI badakŭ ‘duck’ lacks a verified full DEDR family or immediate donor."),
    130: ([], "LSI sūryuḍŭ ‘sun’ appears Indo-Aryan-derived, but its immediate donor and Telugu formation were not verified."),
    131: ([], "LSI candruḍŭ ‘moon’ appears Indo-Aryan-derived, but its immediate donor and Telugu formation were not verified."),
    133: ([], "LSI aggĭ ‘fire’ resembles Sanskrit agni but an immediate donor and Telugu sound adaptation were not verified."),
}

assert set(parents).isdisjoint(specific_holds)
assert set(parents) | set(specific_holds) | set(range(67, 79)) | set(range(80, 104)) == set(range(51, 134))
accepted, held = [], []
for number in range(51, 134):
    record = rows[number - 1]
    assert record["Source"].startswith("grierson-lsi1928")
    if number in parents:
        parent, evidence = parents[number]
        accepted.append(dict(record=record, parent=parent, kind="reflex", citation=record["Source"] + ";dedr[" + parent[1:] + "]", evidence=evidence, status="pending-review"))
    elif number in specific_holds:
        candidates, reason = specific_holds[number]
        held.append(dict(record=record, candidates=candidates, reason=reason))
    else:
        held.append(dict(record=record, candidates=[], reason="This LSI response is a complete subject-plus-predicate clause. A single verbal-root reflex would discard the overt pronoun and tense/person morphology; complete component analysis was not established."))

(P / "proposed-decisions-002.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted, held=held), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds-002.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")

evidence_ids = {parent for parent, _ in parents.values()} | {candidate for candidates, _ in specific_holds.values() for candidate in candidates}
evidence = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt", encoding="utf-8", newline="") as handle:
    for article in csv.DictReader(handle):
        if article["Entry_ID"] in evidence_ids:
            evidence[article["Entry_ID"]] = article
assert set(evidence) == evidence_ids, sorted(evidence_ids - set(evidence))
(P / "primary-evidence-002.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(proposed=len(accepted), held=len(held), evidenceArticles=len(evidence))))
