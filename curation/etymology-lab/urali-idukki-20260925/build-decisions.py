"""Research-only Urali Varghese 2015 residual decisions."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
proposals = {}
reasons = {}

def add(indices, parents, evidence):
    for ix in indices:
        assert ix not in proposals and ix not in reasons
        proposals[ix] = (parents, evidence)

def hold(indices, reason):
    for ix in indices:
        assert ix not in proposals and ix not in reasons
        reasons[ix] = reason

add([3, 4], 5031, "DEDR 5031 includes Tamil mūñci and Malayalam muññi/moññi 'face'; Urali muñḍźi and muñni are locality variants of this face stem.")
add([5], 1977, "DEDR 1977(a) gives Tamil and Malayalam cevi 'ear'; Urali cevicɨ preserves the cevi- ear noun with a local ending.")
add([6], 1448, "DEDR 1448 gives Malayalam kātu and Irula kādu 'ear'; Urali kādɨ is the voiced local ear form.")
add([7, 8], 1591, "DEDR 1591 gives Malayalam ciṟi/ciri 'lip, corner of mouth' and Kodagu kiri 'lip'; both Urali ciri forms have a broader 'mouth' gloss, which needs review.")
add([10, 11], 5409, "DEDR 5409 gives Tamil viral and Malayalam viral 'finger'; Urali veranḍā/viranḍā retain the vir-/ver- finger stem with an extended ending.")
add([20, 21], 1655, "DEDR 1655 gives Tamil kuṭi 'house, abode, town' and Malayalam kuṭi 'house, hut, family'; both Urali kuḍi 'village' records fit this settlement family.")
add([32], 3310, "DEDR 3310 lists Tamil tuṇṭu 'small piece of cloth, towel' and Malayalam tuṇṭam 'piece'; Urali tu nḍu is the cloth noun with source spacing retained.")
add([42, 43], 3045, "DEDR 3045 directly lists Tamil and Malayalam taṇṇi 'water'; both Urali ta nni records preserve that word with source spacing and dental transcription.")
add([48], 2453, "DEDR 2453 lists Tamil cāmpal/cāmpar and Malayalam cāmpal 'ashes'; Urali cāmbalɨ is a direct ash match with local voicing and ending.")
add([66], 3650, "DEDR 3650 directly lists Tamil/Malayalam nāy and Kannada nāyi 'dog'; Urali nāyɨ matches the dog noun.")
add([85], [399, 4198], "Ordered male + child: DEDR 399 gives Tamil āṇ 'male' and 4198 gives Tamil piḷḷai 'child, son'; Urali āmbuḷḷa 'boy' preserves that male-child formation with local assimilation.")
add([86], [4395, 4198], "Ordered female + child: DEDR 4395 gives Tamil peṇ 'woman, girl' and 4198 piḷḷai 'child, daughter'; Urali pembuḷḷa 'girl' preserves both elements.")
add([110, 111], 5496, "DEDR 5496 gives Malayalam veḷḷa 'white' and Tamil veḷ/veṇ; Urali velɨtta dɨ and vella preserve the white stem, one with extra descriptive morphology.")
add([114, 115], 910, "DEDR 910 lists Tamil/Malayalam ēr̤u and Kota e·y 'seven'; Urali eyɨ is especially close to the Kota reduced form.")
add([116, 117], [3918, 990], "Ordered ten + one: DEDR 3918 gives patiṉ- in eleven and 990 gives the one numeral; both Urali padɨnonɨ eleven records preserve this transparent compound.")
add([118, 119], 5151, "DEDR 5151 gives the interrogative e- family with Tamil evaṉ 'which man' and Malayalam āra 'who'; Urali eman 'who?' is a local animate e- interrogative.")
add([120, 121], 5151, "DEDR 5151 lists Tamil eṉṉa and Malayalam entu 'what'; Urali enne/enna are close question-word forms.")
add([122, 123, 124], 5151, "DEDR 5151 includes Tamil eṅku/eṅkē and Malayalam eṅṅu 'where'; Urali eññge/eññgegi retain the nasal-velar place interrogative.")
add([125], 5151, "DEDR 5151 includes Malayalam eppoḷ and Tamil eppōtu 'when'; Urali ebba keeps the ep-/eb- time-interrogative stem with voiced labial.")
add([127, 128], 5151, "DEDR 5151 explicitly lists Malayalam ettira/etra 'how much, how many'; Urali etta ra/ettɨrā preserve both source spellings of that quantifier.")
add([129], 5151, "DEDR 5151 contains Tamil eṉṉa 'what'; Urali ennatta 'what kind?' has that interrogative base with a local suffix.")
add([131, 132], 410, "DEDR 410(a) gives Malayalam itu and Kota id 'this'; both Urali idɨ records are proximal neuter demonstratives.")
add([140], [990, 4597], "Ordered one + like: DEDR 990 gives Malayalam onnu/oru 'one' and 4597 gives Malayalam pōla 'like'; Urali onnɨpola 'same' is literally one-like.")
add([149], 3263, "DEDR 3263 lists Tamil tiṉ and Malayalam tiṉnuka 'eat'; Urali tɨnnɨhe keeps the eat stem with ordinary elicited past morphology.")
add([158], 811, "DEDR 811 gives Malayalam eriyuka and Tamil eri 'burn'; Urali eriyunnu is the corresponding burn verb with progressive morphology.")
add([163, 165], 2426, "DEDR 2426 lists Malayalam cāka with past catt- 'die'; both Urali cattu responses give the past died component of the source's mixed prompt.")
add([170, 171], 3582, "DEDR 3582 gives Tamil naṭa and Malayalam naṭakkuka 'walk'; both Urali naḍannuhe past responses retain the walk stem.")
add([176, 177, 178, 179], 3794, "DEDR 3794 gives Malayalam nōkkuka and Tamil nōkku 'look'; Urali nōkike/nokki are matching elicited past forms in four records.")
add([180, 181], 5160, "DEDR 5160 gives Tamil nāṉ and Malayalam ñān 'I'; Urali nān/nānɨ preserve the Tamil-shaped first-person pronoun.")
add([182, 183], 3684, "DEDR 3684 gives Tamil/Malayalam nī 'you (singular)'; the two Urali ni responses are informal singular.")
add([184, 185], 3688, "DEDR 3688 lists Tamil nīṅkaḷ and Malayalam niṅṅaḷ, also used for respectful singular; Urali niŋgā is explicitly formal singular.")
add([186, 187], 1, "DEDR 1 lists Malayalam avan 'he'; Urali avan is direct and aman shows a local labial-nasal variant requiring review.")
add([188, 189], 1, "DEDR 1 gives Malayalam avaḷ 'she'; Urali avaḷɨ/avaḷārɨ retain the feminine distal base, with respectful or extended ending in the latter.")
add([190, 191], 3647, "DEDR 3647 gives Malayalam nām (oblique nam-) 'we, I and you'; Urali nammā is explicitly inclusive in both locality records.")
add([192, 193], 5154, "DEDR 5154 gives Malayalam ñāṅṅaḷ and Kodagu eŋga 'we' exclusive; Urali yaŋge/yaŋga are explicitly exclusive.")
add([194, 195], 3688, "DEDR 3688 gives Malayalam niṅṅaḷ 'you (plural)'; both Urali niŋga forms are plural, distinct from formal singular niŋgā.")
add([196, 197], 1, "DEDR 1 lists Tamil/Malayalam avar 'they, respectful person'; Urali avammārɨ/avarɨ preserve the distal plural stem with local plural extension.")

hold([1, 2], "deham 'body' is a widespread deha-type loan; Urali's immediate regional donor is not established.")
hold([9, 12, 13, 14, 15], "The elbow, fingernail and leg forms lack a secure exact article match; kākɨ does not straightforwardly preserve the final -l of the kāl 'leg' family.")
hold([16, 17], "karɨppɨ is also an Urali 'black' response; the source's 'skin' gloss conflicts with that colour sense and needs source-level review.")
hold([18, 19], "The urine and feces responses may be euphemistic or inflected; no matching primary family has yet been verified for their exact stems.")
hold([22, 23, 24, 25, 26], "pantā house, panta puratla roof and aḍambɨ door have unresolved house/roof morphology and no secure same-sense comparanda.")
hold([27, 28, 29, 30, 31], "The hammer, axe and needle terms have plausible tool or contact parallels, but their precise family and immediate donor are unresolved; koḍāli should not be assigned merely from superficial spelling.")
hold([33, 34, 35, 36, 37, 38, 39, 40, 41], "Cloth, ring, sun, moon and star forms resemble common regional or Sanskritic vocabulary; the immediate donor path is not yet supported by an attested local source.")
hold([44, 45, 46, 47, 49, 50], "The lightning and rainbow expressions look compound, while śiyāmbarɨ ash and koḷɨkan mud need exact stem comparisons; do not assign one whole-word root by gloss.")
hold([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], "Gold and crop/leaf names circulate through contact; even DEDR tomato parallels do not identify Urali's immediate donor, and cappɨla leaf has uncertain segmentation.")
hold([63, 64, 65, 67, 68, 69, 70, 71], "Cow, buffalo, alternate dog, mosquito and spider names need lexical and donor verification; paṭṭi 'dog' must not be conflated with the paṭṭi 'village' article.")
hold([72, 73, 74, 75, 76, 77, 78, 79], "Man and sibling terms resemble regional or learned kinship vocabulary; immediate donor and sense-specific comparative evidence remain unresolved.")
hold([80, 81, 82, 83, 84], "The son/boy, daughter and husband expressions appear internally complex; kocārɨ has distinct son and boy source records that need joint morphological analysis.")
hold([87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98], "Day-part, week, month and year terms are plausible regional loans or compounds; their immediate sources and internal elements remain unverified.")
hold([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109], "Bad, far, big, heavy and light responses mix loanlike stems and negative/quantity compounds; source segmentation and immediate donors need verification.")
hold([112, 113], "sennḍu 'two' is not a regular match to DEDR 474 iraṇṭu/reṇṭu without explaining the s-initial; hold pending dialect corroboration.")
hold([126, 130], "emblen 'when' and eŋgunaulladɨ 'what kind' are multi-part interrogatives whose endings and stem boundaries are not secure.")
hold([133, 134, 135, 136, 137, 138], "śedɨ 'that' has an unexplained initial ś- relative to DEDR 1, while idellā/adellā combine a demonstrative with an unresolved plural/totalizer element.")
hold([139, 141, 142, 143, 144, 145, 146, 147, 148], "The same/different/whole/broken/few/many responses have uncertain affixes, competing contact analyses, or only broad semantic resemblance; no single inherited stem is secure.")
hold([150, 151, 152, 153, 154, 155, 156, 157], "The hungry and thirsty responses are full clauses with possible belly/water, predicate and tense elements; an isolated root link would flatten the original elicitation.")
hold([159, 160, 161, 162, 164, 166, 167, 168, 169], "Burn, die and kill prompts mix present, past and prohibitive forms; these locality responses need stem and negative/tense segmentation before assignment.")
hold([172, 173, 174, 175], "manḍike/manḍi 'run' is not matched to a secure run family in DEDR; manḍipoyɨ is a serial response with an extra go element.")

for ix, rec in enumerate(records, 1):
    if ix not in proposals and ix not in reasons:
        reasons[ix] = (f"{rec['Form']} ‘{rec['Gloss']}’ remains held pending a secure "
                       "dialect-specific comparandum, donor, or internal segmentation.")

assert set(proposals) | set(reasons) == set(range(1, len(records) + 1))
assert not set(proposals) & set(reasons)
accepted = []
for ix, (parent, evidence) in proposals.items():
    rec = records[ix - 1]
    nums = parent if isinstance(parent, list) else [parent]
    item = {"record": rec, "parent": f"d{nums[0]}",
            "kind": "component" if len(nums) > 1 else "reflex",
            "citation": rec["Source"] + ";" + ";".join(f"dedr[entry {n}]" for n in nums),
            "evidence": evidence, "status": "pending-review", "surveyIndex": ix}
    if len(nums) > 1:
        item["components"] = [f"d{n}" for n in nums]
    accepted.append(item)
held = [{"record": records[ix - 1], "reason": reason, "status": "research-hold",
         "surveyIndex": ix} for ix, reason in reasons.items()]
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
owners = {x["record"]["ID"]: "proposed" for x in accepted}
owners.update({x["record"]["ID"]: "held" for x in held})
assert set(owners) == {r["ID"] for r in records} and len(owners) == len(records)
(P / "record-coverage-001.json").write_text(json.dumps({"source": "varghese-mathew2015idukki",
    "language": "Urali", "residualSourceRecords": len(records), "proposedRecords": len(accepted),
    "heldRecords": len(held), "unexaminedRecords": 0, "uniqueOwnership": True,
    "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(f"{len(accepted)} proposed records, {len(held)} holds, {sum(len(x.get('components', [x['parent']])) for x in accepted)} rows")
