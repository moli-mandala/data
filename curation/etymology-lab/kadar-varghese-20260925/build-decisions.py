"""Research-only Kadar Varghese 2015 residual decisions."""
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

add([2], 5099, "DEDR 5099 gives Tamil/Malayalam mēl 'body' and Kodagu me·lï 'body'; Kadar melɨ is an especially close local form and meaning match.")
add([5, 6], 1448, "DEDR 1448 gives Malayalam kātu and Irula kādu 'ear'; the Kadar kādɨ/kātɨ dialect pair has that ear stem with expected voicing variation.")
add([11], 2023, "DEDR 2023 directly lists Malayalam kai/kayyi and Kota kay 'hand, arm'; Kadar kaya is the same hand/arm noun with a final vowel.")
add([17], 3736, "DEDR 3736 gives Malayalam neñcu/neññu and Tamil neñcam 'heart, chest'; nɛñjakam preserves the neñc- heart stem, with a longer local ending.")
add([20], 4210, "DEDR 4210 directly gives Tamil and Malayalam pī 'excrement' and Kota pi·; Kadar pi is a close form and exact meaning match.")
add([21], 3868, "DEDR 3868 specifically lists Tamil paṭṭi 'hamlet, village' and Kannada paṭṭi 'abode, hamlet'; Kadar paṭṭi matches form and settlement sense.")
add([22, 25], 2007, "DEDR 2007 lists Tamil cēri 'village, hamlet' and Malayalam cēri 'village street'; both Kadar cēri records retain the settlement word, with 'house' a narrower locality gloss.")
add([23, 24], 1904, "DEDR 1904 lists Malayalam kūra 'hut, thatch' and Tamil kūrai 'roof, small hut'; both Kadar kūra 'house' records fit the shelter family.")
add([28], [4867, 4121], "Ordered pepper/chilli + rock: DEDR 4867 lists Malayalam miḷaku and Kota meḷg 'chilli'; 4121 lists Malayalam pāṟa 'rock, large stone'. Kadar miḷakpara names a spice-grinding slab.")
add([35], 4947, "DEDR 4947 lists Malayalam muṇṭu 'short cloth' and Kodagu muṇḍï 'waistcloth'; Kadar muṇḍɨ cloth matches the Kodagu form and sense closely.")
add([43, 44], 3045, "DEDR 3045 explicitly gives Tamil and Malayalam taṇṇi 'water'; the two spaced Kadar ta ṇṇi source records preserve that water noun.")
add([57, 58], 4867, "DEDR 4867 lists Malayalam miḷaku/muḷaku 'pepper' and Kota meḷg 'chilli'; both Kadar mɔḷakɨ/miḷakɨ are regional chilli reflexes within that whole article.")
add([87, 88], 601, "DEDR 601 gives Tamil uṇaṅku and Malayalam uṇaṅṅuka 'become dry'; both Kadar uṇandatɨ records preserve the dry verb with an inflected ending.")
add([89, 90], 3692, "DEDR 3692 gives Tamil nīḷ/nīḷam and Malayalam nīḷam 'length, long'; Kadar niḷamaḷɨ/niḷamāḷi retain that long stem with local adjectival material.")
add([100], 1404, "DEDR 1404 gives Tamil kaṉam 'heaviness' and Malayalam kanakka 'be heavy'; Kadar kanamaḷi retains the heavy adjective stem with local ending.")
add([102, 103], 5496, "DEDR 5496 lists Malayalam veḷḷa and Tamil veḷ/veṇ 'white'; Kadar vɛḷḷa is direct and baḷḷa is a local voiced variant of the same colour stem.")
add([105, 106], 5052, "DEDR 5052 directly lists Tamil mūṉṟu, Malayalam mūnnu and Kota mu·nḏ 'three'; both Kadar mūntɨ forms fit the numeral family.")
add([107, 108], 910, "DEDR 910 lists Tamil/Malayalam ēr̤u and Kota e·y 'seven'; Kadar āyɨ shares the reduced two-vowel form seen in nearby Kota, with vowel quality requiring review.")
add([109, 110], [3918, 990], "Ordered ten + one: DEDR 3918 gives the Dravidian ten numeral and 990 gives Tamil/Malayalam oru 'one'; both Kadar patinɔnnɨ eleven responses retain the transparent numeral formation.")
add([113, 114], 5151, "DEDR 5151 contains Tamil eṉṉa 'what', Malayalam entu 'what', and the wider interrogative e- family; Kadar ɛ̄ṇḍrɨ/ɛndadɨ are corresponding what forms.")
add([115, 116], 5151, "DEDR 5151 includes Malayalam eṅṅu/eṅṅane and Tamil eṅku 'where'; Kadar ɛviḍe/ɛŋgɛ use that interrogative family, with a local place element in ɛviḍe.")
add([117, 118], 5151, "DEDR 5151 lists Malayalam eppoḷ and Tamil eppōtu 'when'; both Kadar ɛppā/ɛppa forms retain the epp- time interrogative.")
add([119, 120], 5151, "DEDR 5151 explicitly includes Malayalam ettira/etra 'how much, how many'; Kadar ɛttra/ɛtta ra correspond closely, preserving both source spellings.")
add([121, 122, 123, 124], 5151, "DEDR 5151 contains Tamil eṉṉa 'what' and Malayalam interrogative eṉṉ-/ent- forms; the four Kadar ɛndɛn/ɛndan 'what kind?' records retain this interrogative stem.")
add([125, 126], 410, "DEDR 410(a) gives Malayalam itu and Kota id 'this'; Kadar itɨ/idɨ reflect the proximal neuter pronoun in both localities.")
add([127, 128], 1, "DEDR 1 gives Tamil/Malayalam atu 'that' and wider distal a- forms; Kadar atɨ/adɨ are matching neuter distal demonstratives.")
add([131], 946, "DEDR 946 lists Malayalam oṭiyuka 'break' and related oṭiccil 'fracture'; Kadar oḍācɨḍɨ 'broken' keeps the oṭ-/oḍ- stem with a local participial ending.")
add([137, 138], 1124, "DEDR 1124 gives Malayalam kaṭikka and Kodagu kaḍi 'bite'; Kadar kaḍin/kaḍi retain that bite stem despite the elicitation's he-bite wording.")
add([151], 2017, "DEDR 2017(a) gives Tamil kēḷ/kēṭṭ- and Malayalam kēḷkka 'hear, listen'; Kadar kēṭɛrɨ preserves the heard past stem with local ending.")
add([152, 153], 3794, "DEDR 3794 explicitly gives Malayalam nōkkuka and Tamil nōkku 'look, see'; both Kadar nokku look records are direct matches.")
add([154, 155], 5160, "DEDR 5160 gives Malayalam ñān 'I'; both Kadar ñān first-person singular responses match directly.")
add([156, 157, 158, 159], 3684, "DEDR 3684 gives Tamil/Malayalam nī 'you (singular)'; Kadar ni/nī retains the same singular stem across informal and formal elicitation slots.")
add([160, 161, 162, 163], 1, "DEDR 1 lists Malayalam avan 'he' and avaḷ 'she'; Kadar avan and avaḷ(ɨ) preserve the masculine and feminine distal pronouns in four records.")
add([164, 165], 3647, "DEDR 3647 directly gives Malayalam nām 'we, I and you'; the two Kadar nām records are explicitly inclusive.")
add([166, 167], 5154, "DEDR 5154 lists Malayalam ñāṅṅaḷ and Kodagu naŋga 'we' exclusive; Kadar naŋkaḷ/ñaŋgaḷ are explicitly exclusive.")
add([168, 169], 3688, "DEDR 3688 lists Tamil plural nīm/nīr and its oblique num-; Kadar nōm/nūm second-person plural responses fit that pronominal family, with rounded vowel requiring review.")
add([170, 171], 1, "DEDR 1 includes Tamil/Malayalam avar/avai 'they'; Kadar abɨ/avɨ are short distal plural responses with local labial variation.")

hold([1, 3, 4], "The source gives divergent body/hair responses; reːgam resembles a regional Sanskritic body word, while taḍi and ta lanarɨ require separate segmentation and donor checks.")
hold([7, 8, 9, 10], "The two koṭṭi mouth records and pullari/pira li tongue responses lack a secure matching primary article; their near homophones have different tool or motion meanings.")
hold([12, 13, 14, 15], "The elbow and palm responses appear compound or descriptive (notably final kai 'hand'); the initial elements are not securely identified, so whole-word reflexes are withheld.")
hold([16, 18, 19], "Nagam 'nail' resembles a regional nakha loan; uyirɨkuṇḍaka 'heart' is multi-part; aṭṭal 'urine' lacks a secure local comparandum.")
hold([26, 27, 29, 30], "The roof, mortar and pestle responses contain recognizable shelter/stone elements, but their other elements and precise tool segmentation remain unresolved.")
hold([31, 32, 33, 34], "The hammer terms have uncertain tool senses, while koḍāli 'axe' has contact and inheritance comparisons; the immediate donor or exact family is unresolved.")
hold([36, 37, 38, 39, 40], "Ring, sun and sky responses resemble widespread regional or learned vocabulary; an immediate donor has not been established for these Kadar attestations.")
hold([41, 42, 45, 46, 47, 48], "The star, lightning and rainbow responses are not transparently one stem; manivɨ and vāḷminnaḍɨ may contain a light/flash element but full lexical segmentation remains uncertain.")
hold([49, 50, 51, 52, 53, 54], "Ash cāram/cāramaṇṇɨ, gold sorṇa and wheat kōta mbi/gōdambɨ present possible contact forms or compounds; immediate donors and first elements remain unverified.")
hold([55, 56, 59, 60, 61, 62, 63, 64], "Potato, eggplant, onion, tomato and cabbage are crop names with broad regional circulation; even where a DEDR article lists a parallel (tomato under 3009), it does not settle Kadar's immediate borrowing path.")
hold([65, 66, 67, 68, 69], "Mosquito, spider and child responses lack a secure form-and-meaning match in checked primary entries; no etymon is assigned by gloss alone.")
hold([70, 71, 72, 73], "The sibling labels acci, aniyatti and paṇkaḷ have competing kinship interpretations; DEDR ācci names mother/grandmother, not directly an elder sister here.")
hold([74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84], "The morning/noon/week/month/year expressions mix possible contact terms and compounds; immediate donor and internal morphology need individual verification.")
hold([85, 86, 91, 92, 93, 94], "The bad, short and cold adjective responses require stem and suffix analysis; cītta resembles a regional śīta-type contact word whose donor is unresolved.")
hold([95, 96, 97, 98, 99, 101, 104], "The near/far/small/below/red expressions include descriptive morphology or possible dūra-type borrowing; form-specific ancestry remains uncertain.")
hold([111, 112], "ɛ̄tɨ is glossed 'who?' but the closest DEDR 5151 ētu/etu members mean 'what, which'; this question-word mismatch needs dialect corroboration.")
hold([129, 130], "oreːmātiri 'same' is a multi-part one-plus-manner expression; the mātiri element may be borrowed and needs an immediate donor before ordered components are assigned.")
hold([132, 133, 134, 135, 136], "The few/many quantifiers have competing etymologies or unclear local morphology; a semantic resemblance to a generic small/large family is insufficient.")
hold([139, 140, 141, 142], "The hungry and thirsty responses are full predicative clauses: paṭṭiṇi is recognizable as 'starvation' (DEDR 3872), but the remaining verb and tense elements require analysis.")
hold([143, 144, 145, 146, 147, 148], "The burn/die/kill prompts mix present, past and prohibitive meanings; each response needs verb-stem and negative/tense segmentation before a root assignment.")
hold([149, 150], "The fly response has an uncertain stem, and ōḍi ōḍi pō 'run' is a serial-verb phrase rather than a single inherited lexical item.")

# Keep any remaining case visible as an individual hold.
for ix, rec in enumerate(records, 1):
    if ix not in proposals and ix not in reasons:
        reasons[ix] = (f"{rec['Form']} ‘{rec['Gloss']}’ remains held: its immediate donor, "
                       "source segmentation, or exact lexical family is not established by the checked evidence.")

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
(P / "record-coverage-001.json").write_text(json.dumps({"source": "varghese2015palakkad",
    "language": "Kadar", "residualSourceRecords": len(records), "proposedRecords": len(accepted),
    "heldRecords": len(held), "unexaminedRecords": 0, "uniqueOwnership": True,
    "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(f"{len(accepted)} proposed records, {len(held)} holds, {sum(len(x.get('components', [x['parent']])) for x in accepted)} rows")
