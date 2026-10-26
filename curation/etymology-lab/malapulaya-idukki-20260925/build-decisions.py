"""Draft source-preserving Mala Pulaya proposals; no accepted overlay writes."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())

# One-based indexes preserve repeated source-defined records. DEDR citations
# refer to whole articles, verified against the full cached dictionary prose.
proposals = {
    2: (1448, "DEDR 1448 lists Malayalam kātu 'ear' and Irula kādu; Mala Pulaya kātɨ matches that ear noun."),
    6: (4210, "DEDR 4210 gives Malayalam pī and Telugu pīyi/piyyi 'excrement'; the survey piyɨ has the same p-y shape and feces sense."),
    11: ([184, 1298], "Ordered grinding-stone + stone: DEDR 184 gives Malayalam ammi 'grinding stone', and 1298 gives Malayalam kal/kallu 'stone'. The full amikallɨ form is retained despite the source's broader 'mortar' gloss."),
    17: (3045, "DEDR 3045 explicitly lists Malayalam taṇṇi and Tamil taṇṇi 'water'; the survey's spaced ta ṇṇi denotes that water noun."),
    34: (4411, "DEDR 4411 gives Malayalam and Tamil periya 'large' and the southern per-/peri- family; survey pɛriśɨ is a local big adjective with altered ending."),
    37: (5496, "DEDR 5496(a) gives Malayalam veḷḷa 'white colour' and a wide veḷ-/veṇ- white family; survey vaḷḷa matches its consonants and sense."),
    39: (5151, "DEDR 5151 directly includes Malayalam/Tamil ār 'who'; the survey ārɨ preserves that interrogative with a final vowel."),
    40: (5151, "DEDR 5151 gives Tamil eṉṉa 'what' and Malayalam entu; survey ɛnna is the e-n-n interrogative form."),
    41: (5151, "DEDR 5151 groups Malayalam eṅṅu and Tamil eṅkē 'where'; survey ɛŋgɛ preserves the same interrogative-place formation."),
    42: (5151, "DEDR 5151 includes Tamil eppōtu and Malayalam eppōḷ 'when'; survey ɛppa belongs to that temporal interrogative family."),
    43: (5151, "A second source-defined ɛppa 'when' record is retained separately; DEDR 5151's epp- temporal interrogatives support the same family."),
    44: (5151, "DEDR 5151 explicitly gives Tamil evvaḷavu 'how much' and related e-/ev- quantity questions; survey ɛvaḷavɨ closely matches."),
    46: (410, "DEDR 410(a) lists Malayalam itu 'this thing' within the proximal i- paradigm; survey itɨ matches the neuter demonstrative."),
    47: (1, "DEDR 1 lists Malayalam atu 'that thing' within the distal a- paradigm; survey atɨ matches the neuter demonstrative."),
    48: (410, "The survey repeats itɨ for 'these'; DEDR 410(a) includes the whole i- proximal paradigm. Number is not overt in this survey form, so the source gloss is retained."),
    49: (1, "The survey repeats atɨ for 'those'; DEDR 1 includes the whole a- distal paradigm. Number is not overt in this survey form, so the source gloss is retained."),
    57: (1124, "DEDR 1124 gives Malayalam kaṭikka 'bite' and Tamil kaṭi; kaḍiciṭa retains the bite stem with ordinary past elicitation morphology."),
    60: (1654, "DEDR 1654 gives Malayalam kuṭikka and Kodagu kuḍi- 'drink'; kuḍiśacɨ retains that drink stem in a single inflected response."),
    63: (3376, "DEDR 3376(a) gives Malayalam tūṅṅuka and Tamil tūṅku 'sleep'; the survey tu ːŋgiṭa retains the sleep stem with past morphology."),
    64: (3852, "DEDR 3852 gives Malayalam paṭuka (past paṭṭ-) 'lie down'; paḍattiṭa is a comparable inflected elicitation response."),
    65: (480, "DEDR 480 gives Tamil iru (past irunt-) and Malayalam irikkuka 'sit'; irɨndiṭa retains the irund- past stem."),
    66: (3098, "DEDR 3098 gives Tamil taru (past tant-) and Malayalam taruka/tarika (past tann-) 'give'; spaced ta nta cɨ retains the giving past stem."),
    67: (811, "DEDR 811 gives Tamil eri and Malayalam eriyuka 'burn'; ɛriyatɨ retains that burn stem in an inflected response."),
    77: (3582, "DEDR 3582 lists Malayalam naṭakka 'walk' and Tamil naṭa; naḍandacɨ is a single past walk predicate preserving naḍ-/naṭ-."),
    78: (1041, "DEDR 1041 gives Tamil ōṭu (ōṭi-) and Malayalam ōṭuka 'run'; ōḍiṭa has the running stem with ordinary past morphology."),
    79: (4572, "DEDR 4572 gives Malayalam pōka 'go' and Tamil pō; poyiricɨ contains the go stem with source past morphology."),
    80: (5270, "DEDR 5270 gives Malayalam varuka (past vann-) and Tamil varu (past vant-) 'come'; vandacɨ preserves the past come stem."),
    81: (2855, "DEDR 2855 gives Malayalam colluka and Tamil col 'say, speak'; colliyac. retains the speaking stem and source punctuation."),
    82: (2017, "DEDR 2017(a) gives Tamil kēḷ (past kēṭṭ-) and Malayalam kēḷkka 'hear, listen'; kɛṭac. preserves the heard past stem."),
    83: (4091, "DEDR 4091(a) gives Tamil pār 'look' and Malayalam pārkka 'behold'; pārɨ is the same look stem."),
    85: (5160, "DEDR 5160 gives Tamil nāṉ and cognate first-person singular pronouns, while Malayalam has ñān; survey nān shares the Tamil-shaped singular form."),
    86: (3684, "DEDR 3684 lists Tamil/Malayalam nī 'you (sg.)'; survey ni is the informal singular pronoun."),
    87: (3688, "DEDR 3688 lists Malayalam niṅṅaḷ 'you (pl.)', also used for respectful address; survey niŋgɛ has a formal singular source gloss."),
    88: (1, "DEDR 1 explicitly gives Malayalam avan 'that man, he'; survey avan matches the masculine distal pronoun."),
    89: (1, "DEDR 1 gives Malayalam avaḷ 'that woman, she'; survey ava is a shorter feminine distal form within this paradigm."),
    90: (3647, "DEDR 3647 gives Malayalam nām (oblique nam-) 'we, I and you'; survey namma is explicitly inclusive."),
    91: (5154, "DEDR 5154 gives Malayalam ñāṅṅaḷ exclusive 'we' and Kodagu naŋga; survey naŋga is explicitly exclusive and matches the latter shape."),
    92: (3688, "DEDR 3688 lists Malayalam niṅṅaḷ 'you (pl.)'; survey niŋgɛ in the plural slot belongs to this paradigm, separate from the formal-singular record."),
    93: (1, "DEDR 1 gives Malayalam avar 'those persons, they'; survey avarɨ has the same distal plural pronoun stem."),
}

reasons = {
    1: "mɔkham 'face' appears to be a mukha-type regional loan; immediate donor not verified.",
    3: "nakam 'fingernail' resembles nakha/nakham contact vocabulary; immediate donor is unresolved.",
    4: "kola 'heart' lacks a form-and-sense match in the checked DEDR heart families.",
    5: "mōḷɨ 'urine' lacks a secure comparative parent; a superficial mūḷ- match is not enough.",
    7: "kra ːmam 'village' is grāma-type contact vocabulary; immediate donor unresolved.",
    8: "ūḍɨ 'house' may relate to vīṭu (DEDR 5393), but its initial-vowel shape needs dialect-specific corroboration.",
    9: "cimarɨ 'broom' lacks a secure first-hand lexical comparandum.",
    10: "viḷakkamarɨ 'broom' appears compound; both elements and the whole sense require confirmation.",
    12: "koḷavikallɨ 'pestle' has clear kallɨ 'stone' but the first element and tool construction are unverified.",
    13: "koḍali 'axe' has DEDR Appendix 32 and CDIAL 3244 comparisons; immediate loan donor and direction remain unresolved.",
    14: "motiram 'ring' resembles a widespread loan (CDIAL 10203); immediate regional donor unresolved.",
    15: "sūriyan 'sun' is sūrya-type learned/contact vocabulary; immediate donor unresolved.",
    16: "tsandaran 'moon' is candra-type learned/contact vocabulary; immediate donor unresolved.",
    18: "tsonavillɨ 'rainbow' appears compound with villɨ 'bow'; first element not securely analysed.",
    19: "kota mbɨ 'wheat' is a contact crop term with uncertain donor and source spacing.",
    20: "īravɛŋgayam 'onion' is a compound regional crop name; components and ancestry need checking.",
    21: "ta kkāḷi 'tomato' is a post-Columbian crop name with uncertain regional borrowing path.",
    22: "mɔṭakośɨ 'cabbage' is a regional crop name with uncertain immediate donor.",
    23: "purśan 'husband' resembles puruṣa-type contact vocabulary; no verified immediate donor.",
    24: "First aṇpɔḷapayyan 'boy' record combines male, ethnic/community and boy terms; the middle segment is unresolved.",
    25: "Second source-defined aṇpɔḷapayyan 'boy' record; incomplete compound analysis.",
    26: "pɛṇpuḷapiḷḷa 'girl' is a multi-part response whose middle community modifier needs analysis.",
    27: "rātri 'night' is rātri-type contact vocabulary; immediate donor unresolved.",
    28: "matyanam 'noon' appears madhyāhna-type borrowed time vocabulary; immediate donor unresolved.",
    29: "sāyamkalam 'evening/afternoon' has time-of-day components requiring separate ancestry and segment verification.",
    30: "varam 'week' is regional calendar vocabulary; immediate donor unresolved.",
    31: "masam 'month' is likely māsa-type borrowed calendar vocabulary; immediate donor unresolved.",
    32: "varśam 'year' is likely varṣa-type borrowed calendar vocabulary; immediate donor unresolved.",
    33: "dūram 'far' resembles Indo-Aryan dūra; immediate donor unresolved.",
    35: "First cariyanakanam 'heavy' response appears phrasal, perhaps ending in kanam; segmentation remains unclear.",
    36: "Second source-defined cariyanakanam 'heavy' record; whole response requires analysis.",
    38: "patinoṇṇɨ 'eleven' is transparently ten-plus-one, but exact component targets and inflection need verification.",
    45: "ɛnnaśayis 'what kind?' includes an interrogative and an uncertain kind/way element; complete analysis unavailable.",
    50: "orēmatiri 'same' combines orē (DEDR 990) and a likely contact manner word; second component donor unresolved.",
    51: "palaśayis 'different' includes a probable 'many' element but remaining morphology is uncertain.",
    52: "First kanama 'many' record lacks a secure comparable quantitative stem.",
    53: "Second source-defined kanama 'many' record lacks a secure comparable quantitative stem.",
    54: "pūra 'all' resembles regional pūrā loan vocabulary; immediate donor unresolved.",
    55: "cappiḍɨ 'eat' is a cāppiṭu-type verb not securely placed in DEDR; contact history uncertain.",
    56: "cappiṭacɨ 'he ate' continues the same uncertain cāppiṭu-type verb; keep distinct past record.",
    58: "paśikkatɨ 'he is hungry, was hungry' resembles Tamil paci, but the survey stem and mixed tense elicitation need further analysis.",
    59: "paśiyaricɨ 'he is hungry, was hungry' may combine hunger with an auxiliary; no whole-word stem link.",
    61: "ta vikkatɨ 'he is thirsty, was thirsty' is a spaced predicate with unresolved lexical stem.",
    62: "ta hamayiric 'he is thirsty, was thirsty' is a distinct complex predicate with unresolved components.",
    68: "First cakakuḍatɨ record mixes 'don't die' and 'he died'; negative and verb morphology need separation.",
    69: "First cattu pona record is a two-verb response under contradictory negative/past gloss; do not flatten it.",
    70: "Second source-defined cakakuḍatɨ record retains separate identity and unresolved negative/past analysis.",
    71: "Second source-defined cattu pona record retains separate identity and compound verbal analysis.",
    72: "cakavaykakuḍatɨ 'don't kill, he killed' contains negation and a complex kill predicate.",
    73: "cakavacɨya 'don't kill, he killed' is another kill elicitation with unresolved morphology.",
    74: "parannapo 'fly, it flew' combines fly and go stems; ordered full analysis is needed.",
    75: "paradupoyirɨcɨ 'fly, it flew' is a longer compound predicate; do not link only its fly/go root.",
    76: "naḍandupo 'walk, he walked' combines a walk participle and go; two-component analysis is needed.",
    84: "pattā c 'look, he looked' may represent pār- past but the recorded form is too divergent to assign confidently.",
}

assert set(proposals) | set(reasons) == set(range(1, len(records) + 1))
assert not set(proposals) & set(reasons)
accepted = []
for ix, (number, evidence) in proposals.items():
    rec = records[ix - 1]
    nums = number if isinstance(number, list) else [number]
    item = {"record": rec, "parent": f"d{nums[0]}",
            "kind": "component" if len(nums) > 1 else "reflex",
            "citation": rec["Source"] + ";" + ";".join(f"dedr[entry {n}]" for n in nums),
            "evidence": evidence, "status": "pending-review", "surveyIndex": ix}
    if len(nums) > 1:
        item["components"] = [f"d{n}" for n in nums]
    accepted.append(item)
held = [{"record": records[ix - 1], "reason": reason,
         "status": "research-hold", "surveyIndex": ix} for ix, reason in reasons.items()]
accepted.sort(key=lambda x: x["surveyIndex"])
held.sort(key=lambda x: x["surveyIndex"])
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted,
    "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
parents = {pid for x in accepted for pid in x.get("components", [x["parent"]])}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
source_ids = [r["ID"] for r in records]
owners = {x["record"]["ID"]: "proposed" for x in accepted}
owners.update({x["record"]["ID"]: "held" for x in held})
assert set(owners) == set(source_ids) and len(owners) == len(source_ids)
(P / "record-coverage-001.json").write_text(json.dumps({"source": "varghese-mathew2015idukki",
    "language": "MalaPulaya", "residualSourceRecords": len(records),
    "proposedRecords": len(accepted), "heldRecords": len(held), "unexaminedRecords": 0,
    "uniqueOwnership": True, "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(f"{len(accepted)} proposed records, {len(held)} holds, {sum(len(x.get('components', [x['parent']])) for x in accepted)} rows")
