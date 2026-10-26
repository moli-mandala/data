"""Assess the remaining 101 Kui LSI records; pending review only."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "inventory.json").read_text())
assert len(records) == 151

question = ("d5151", "DEDR 5151's full Kui interrogative family includes imbai ‘who’, in- ‘what’, and inaki ‘why’. The LSI form belongs to that printed family; the survey gloss may be broader than the cited subform.")
we = ("d5154", "DEDR 5154 gives Kui āju inclusive ‘we’, āmu exclusive ‘we’, and oblique mā-. The LSI form is one of those subject or possessive variants.")
demonstrative = ("d764", "DEDR 764 gives Kui eanju ‘that man’, earu ‘those men’, and related e-/ē- forms. The LSI ē- nominative or possessive form belongs to this nearer-demonstrative family.")
proposals = {
    53: ("d5151", "DEDR 5151 explicitly gives Kui imbai ‘who’; LSI imbāi matches with the survey's long vowel."),
    55: question,
    56: question,
    57: question,
    60: ("d333", "DEDR 333 gives Kui ā, āngo ‘yes’; LSI a is the affirmative with unmarked vowel length in the survey."),
    61: ("d333", "DEDR 333 explicitly gives Kui āë [ā'e] ‘no’; LSI āē is the same negative answer in survey spelling."),
    64: ("d1628", "DEDR 1628 gives Kui kopka from underlying kok-p-/kokt- ‘sit’; LSI kok is the unsuffixed stem."),
    68: ("d5160", "DEDR 5160 explicitly gives Kui ānu ‘I’; LSI ānu matches it."),
    69: ("d5160", "DEDR 5160 gives Kui ānu with oblique nā-; LSI nā ‘my’ matches that possessive stem."),
    70: ("d5160", "DEDR 5160 gives Kui ānu with oblique nā-; LSI nai ‘my’ is an inflected/phonetic variant of that possessive stem."),
    71: we,
    72: we,
    74: we,
    76: ("d3684", "DEDR 3684 gives Kui īnu/nīnu with oblique nī-; LSI nī ‘thy’ matches the possessive stem."),
    77: ("d3688", "DEDR 3688 explicitly gives Kui īru ‘you’; LSI īru matches it."),
    79: ("d3675", "DEDR 3675 gives Kui nilpa from nilt- ‘stand’; LSI nil is the unsuffixed elicited stem."),
    80: ("d2598", "DEDR 2598 gives Kui sīva (sīt-) ‘give’; LSI si is the reduced elicited stem in survey transcription."),
    120: ("d3688", "DEDR 3688 gives Kui īru with oblique mī-; LSI mī ‘your’ matches the possessive stem."),
    121: demonstrative,
    122: demonstrative,
    123: demonstrative,
    124: demonstrative,
    128: ("d1977", "DEDR 1977 explicitly gives Kui kriu (plural krīka) ‘ear’; LSI kriu matches it."),
    131: ("d2967", "DEDR 2967 gives Kui ṭūṭu ‘stomach, abdomen, belly’; LSI ṭuṭu matches with shorter survey vowels."),
    147: ("d4705", "DEDR 4705 gives Kui meo ‘cat’; LSI mēõ is its nasalized survey spelling."),
}
component_proposals = {
    151: (["d3754", "d4438"], "The first full article gives Kui ḍānju ‘moon, month’; the second gives Kui pēnu ‘god, spirit’. LSI ḍañjupēnu is their moon-plus-deity compound in that order, with survey vowel and nasal spelling. The source gloss ‘Moon’ names the referent, not a single inherited stem."),
}

holds = {
    51: ([], "LSI muhũtā ‘before’ has no verified full DEDR Kui reflex or immediate donor."),
    52: ([], "LSI bētoṭi ‘behind’ has no verified full DEDR Kui reflex or immediate donor."),
    54: (["d5151"], "LSI umbāi ‘who’ may be a dialectal variant of DEDR 5151 imbai, but the u-/i- difference is not supported by the full article."),
    58: (["d410"], "LSI oṭē ‘and’ lacks a verified source; DEDR 410 gives Kui enga ‘and’, not this form."),
    59: ([], "LSI gē ‘but’ lacks a verified full DEDR Kui family or immediate donor."),
    62: (["d333"], "DEDR 333 gives Kui āë ‘no’, but LSI siḍē is not that form and its own source remains unresolved."),
    63: ([], "LSI sāl ‘go’ has no direct full DEDR Kui match; the article forms found by gloss have other stems."),
    65: ([], "LSI kaḍē ‘twenty’ needs numeral morphology and a verified family; no direct Kui DEDR form was found."),
    66: ([], "LSI ri kaḍi daśu ‘fifty’ is a multiword numeral phrase with unsourced members, not one inherited numeral stem."),
    67: ([], "LSI pā̃su kaḍi ‘hundred’ is a compound numeral phrase with unsourced members."),
    73: (["d5154"], "LSI ammāni ‘our’ may contain the DEDR 5154 mā- oblique stem, but the whole formation needs analysis."),
    75: (["d5154"], "LSI mai ‘our’ may belong with DEDR 5154 mā-; the diphthongal survey form is not established by the article."),
    78: (["d5270"], "DEDR 5270 gives Kui vāva, imperative va ‘come’; LSI bā has a different initial consonant, whose dialect or transcription status is unverified."),
    81: (["d4145A"], "DEDR 4145A gives Kui pinja ‘rebound, leap’, whereas LSI pinjā is glossed ‘run’; the sense extension requires confirmation."),
    94: ([], "LSI sāh ‘beat’ has no verified full DEDR Kui family or immediate donor."),
    95: ([], "LSI uh ‘beat’ has no verified full DEDR Kui family or immediate donor."),
    125: ([], "LSI pātkā ‘foot’ lacks a verified full DEDR Kui family or immediate donor."),
    126: ([], "LSI suaā ‘mouth’ lacks a verified full DEDR Kui family or immediate donor."),
    127: ([], "LSI pālkā ‘tooth’ lacks a verified full DEDR Kui family or immediate donor."),
    129: ([], "LSI lēñjā ‘hair’ lacks a verified full DEDR Kui family or immediate donor."),
    130: ([], "LSI bāŋgōsi ‘tongue’ lacks a verified full DEDR Kui family or immediate donor."),
    132: ([], "LSI bēta ‘back’ lacks a verified full DEDR Kui family or immediate donor."),
    133: ([], "LSI luhā ‘iron’ resembles Indo-Aryan loha, but the immediate donor into Kui was not verified."),
    134: ([], "LSI sunā ‘gold’ resembles Indo-Aryan sonā, but the immediate donor into Kui was not verified."),
    135: ([], "LSI dārbā ‘silver’ lacks a verified full DEDR Kui family or immediate donor."),
    136: ([], "LSI āliā ‘slave’ lacks a verified full DEDR Kui family or immediate donor."),
    137: ([], "LSI cāsāgāṭāñju ‘cultivator’ is a derived or compound person term whose members have not been securely identified."),
    138: ([], "LSI gāuḍēñju ‘shepherd’ is a derived or compound person term whose members have not been securely identified."),
    139: (["d4438"], "DEDR 4438 gives Kui pēnu ‘god’, but LSI roṭāpēŋu contains an unsourced initial modifier; the whole compound cannot be assigned to the deity noun alone."),
    140: ([], "LSI osurēñju ‘devil’ is an opaque person/deity formation without a verified full family or donor."),
    141: ([], "LSI sirō ‘water’ lacks a verified full DEDR Kui family or immediate donor."),
    142: ([], "LSI kākēri ‘water’ lacks a verified full DEDR Kui family or immediate donor."),
    143: ([], "LSI gōḍā ‘horse’ lacks a verified inherited family or immediate donor; a possible Indo-Aryan relation was not proven."),
    144: ([], "LSI goḍo ‘ass’ lacks a verified inherited family or immediate donor; a possible Indo-Aryan relation was not proven."),
    145: ([], "LSI ōṭo ‘camel’ lacks a verified full DEDR Kui family or immediate donor."),
    146: (["d1438"], "DEDR 1438 has Kui kāsi nakuri ‘wild dog’, but does not establish LSI nākuaḷi ‘dog’ as a whole or give its exact stem."),
    148: ([], "LSI ṭādrā koju ‘cock’ is a two-part expression without a complete sourced component analysis."),
    149: ([], "LSI honso ‘duck’ lacks a verified full DEDR Kui family or immediate donor."),
    150: (["d4438"], "DEDR 4438 supplies pēnu ‘god’, but the initial bēlā in LSI bēlāpēnu ‘sun’ has no verified parent; the whole compound remains held."),
}

for number in list(range(82, 94)) + list(range(96, 120)):
    holds[number] = ([], "This LSI response is a complete subject-plus-predicate clause. Linking it to one pronoun or verbal root would lose the overt other member and inflection; a complete component analysis was not established.")

assert set(proposals).isdisjoint(component_proposals)
assert set(proposals).isdisjoint(holds)
assert set(component_proposals).isdisjoint(holds)
assert set(proposals) | set(component_proposals) | set(holds) == set(range(51, 152))
accepted, held = [], []
for number in range(51, 152):
    record = records[number - 1]
    assert record["Source"].startswith("grierson-lsi1928")
    if number in proposals:
        parent, evidence = proposals[number]
        accepted.append(dict(record=record, parent=parent, kind="reflex", citation=record["Source"] + ";dedr[" + parent[1:] + "]", evidence=evidence, status="pending-review"))
    elif number in component_proposals:
        parts, evidence = component_proposals[number]
        accepted.append(dict(record=record, parent=parts[0], components=parts, kind="component",
                             citation=record["Source"] + "".join(";dedr[" + part[1:] + "]" for part in parts), evidence=evidence,
                             status="pending-review"))
    else:
        candidates, reason = holds[number]
        held.append(dict(record=record, candidates=candidates, reason=reason))
(P / "proposed-decisions-002.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted, held=held), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds-002.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")

evidence_ids = {v[0] for v in proposals.values()} | {part for parts, _ in component_proposals.values() for part in parts} | {candidate for candidates, _ in holds.values() for candidate in candidates}
evidence = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt", encoding="utf-8", newline="") as handle:
    for article in csv.DictReader(handle):
        if article["Entry_ID"] in evidence_ids:
            evidence[article["Entry_ID"]] = article
assert set(evidence) == evidence_ids, sorted(evidence_ids - set(evidence))
(P / "primary-evidence-002.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(proposed=len(accepted), assignmentRows=sum(len(x.get("components", [x["parent"]])) for x in accepted), held=len(held), evidenceArticles=len(evidence))))
