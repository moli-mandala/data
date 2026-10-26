"""Research-only Gondi Varghese 2015 residual decisions."""
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

add([4], 4411, "DEDR 4411 lists Gondi (Ko.) beriya and (D./L.) behra/pehro 'big'; DravLex berra fits that Gondi b/p-r big stem.")
add([13], 1159, "DEDR 1159(a) lists Gondi (A.) kaṛ 'eye' beside other Gondi kan; kəḍə has a comparable initial k- and retroflex consonant, but its final vowel and stop transcription need review.")
add([17], 3682, "DEDR 3682 directly lists Gondi (Ko.) ninḍ- and (Tr./Ph.) nindānā 'be filled'; DravLex ninḍā 'full' retains the same nind-/ninḍ- stem.")
add([18], 2598, "DEDR 2598 lists Gondi (Ma./Ko.) ī- 'give'; īdatam preserves that give stem with a longer source verb ending.")
add([22], 2017, "DEDR 2017(a) lists Gondi kēnj-/kēnjānā 'hear'; DravLex kēja is a shorter hear response with the same kēj-/kēnj- material, and nasal loss needs review.")
add([41], 5058, "DEDR 5058 directly gives Gondi (A./D./Ko.) meṭṭa 'hill' and related meṭa mountain; DravLex meṭṭa 'mountain' matches that locality form.")
add([46], 2559, "DEDR 2559 lists Gondi (Ma.) ille and other hille/sille variants 'not, no'; DravLex ille matches directly.")
add([47], 990, "DEDR 990 gives Gondi (Y.) oror 'one (masculine)' and (Mand.) oṛe; DravLex orrōṭ 'one' keeps the or-/oṛ- numeral with a local ending.")
add([50], 1695, "DEDR 1695 lists Gondi (M.) gonḍra and (ASu.) guṇḍāl 'round'; DravLex guṇḍrān is a close round-adjective reflex.")
add([51], 2017, "DEDR 2017(b) directly lists Gondi (Ko.) kel- and KoyaSu kell- 'tell'; DravLex kella 'say' belongs to that speech subsection of the complete article.")
add([54, 55], 2594, "DEDR 2594 gives Telugu cinna and neighbouring Kolami sinnam 'small', plus Gondi (Mu.) cini in 'little finger'; DravLex cinna/cinnad generalize that small stem, with Telugu contact possible.")
add([57], 4559, "DEDR 4559 lists Gondi poṛd 'sun', Konda podu and Telugu poddu; DravLex poḍudu shares the sun/time family, though its Telugu-like surface shape may reflect contact.")
add([61, 62], 1, "DEDR 1 gives Gondi ad/had 'that woman or thing' and the wider distal a- base; DravLex adu/addu are matching distal neuter forms.")
add([63], 410, "DEDR 410(a) lists the Gondi proximal i- family, and the homepage's existing Gondi iddu is linked there; DravLex iddu 'this' preserves that proximal base.")
add([64], 3986, "DEDR 3986 lists Gondi pal 'tooth' and palk/palkʰ 'teeth'; DravLex palk 'tooth' is a close attested plural-stem variant.")
add([68], 5154, "DEDR 5154 gives Gondi (Mu.) mamma and many mammāṭ variants for 'we'; DravLex mamma matches directly, though the source does not specify clusivity.")
add([69], 5151, "DEDR 5151 lists Gondi bad/badu and bāta 'what'; DravLex bāta is an exact question-word match within the Gondi subsection.")
add([70], 5151, "DEDR 5151 lists Gondi (Ko.) benonḍ and Koi benōṇḍu 'who'; DravLex bēnonḍu matches that animate interrogative closely.")

hold([1, 2], "antā and bottig 'all' have possible regional quantifier or demonstrative analyses, but their exact Gondi ancestry is not established by a primary article.")
hold([3, 5, 6, 7], "Bark, bird and breast responses lack a secure form-and-meaning match; dudo has repeated Gondi records but no accepted same-sense parent.")
hold([8, 9, 10, 11, 12], "The cold, die, dry and eat forms have uncertain stems or contact histories; poḍi resembles a regional dry/powder form but its immediate donor is unresolved.")
hold([14, 15, 16], "gerk feather and neggaḍ fire lack a secure primary cognate; homepage tulla comparisons concern boring/perforation rather than 'fly'.")
hold([19, 20, 21], "Good/green responses lack sound-and-sense support; hī matches a documented Gondi 'give' stem (DEDR 2598), not the source's 'green' gloss.")
hold([23, 24, 25, 26, 27], "Horn, kill, knee and know forms need exact dialectal comparanda; telusē resembles Telugu 'know', but its immediate contact path is unresolved.")
hold([28, 29, 30], "The lie and liver forms have no secure matching Gondi parent; homepage unja candidates concern a hair-knot, not lying down.")
hold([31, 32], "poḍavuṭādu 'long' resembles Telugu poḍavu (DEDR 4484) but is a longer predicate, while lām may be a regional loan; immediate donor and morphology are unresolved.")
hold([33, 34, 35], "The three man forms have distinct apparent stems; pēkāl is matched by DEDR 3939 Gondi pēkal 'boy', but the DravLex 'man' gloss needs clarification before linking.")
hold([36, 37, 38, 39, 40], "Many, moon and mountain gubal forms have no secure same-sense DEDR match; jāsta is an apparent regional loan whose immediate donor is unknown.")
hold([42, 43, 44, 45], "Mouth and neck forms lack secure dialect-specific comparanda; kottad resembles a regional 'new' form but donor direction is unresolved.")
hold([48, 49], "The person responses may contain learned or contact material; no immediate donor is verified.")
hold([52, 53], "ūḍa 'see' conflicts with the homepage's near-form sit article DEDR 725; pēṇḍku 'seed' conflicts with a near-form god/idol article DEDR 4438.")
hold([56, 58, 59, 60], "rāy 'stone' resembles Telugu rāyi (DEDR 321) without a verified immediate borrowing path; swim responses need secure same-sense families.")
hold([65, 66, 67], "naḍam 'walk' and vēḍ 'warm' have unresolved exact Gondi matches; manāḍu 'we' cannot be assigned to inclusive or exclusive article from the source gloss alone.")
hold([71, 72, 73, 74, 75], "The woman and yellow terms have uncertain source stems or contact paths; form resemblance alone does not justify an inherited article assignment.")

for ix, rec in enumerate(records, 1):
    if ix not in proposals and ix not in reasons:
        reasons[ix] = (f"{rec['Form']} ‘{rec['Gloss']}’ remains held pending "
                       "dialect-specific primary evidence or an immediate donor.")

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
(P / "record-coverage-001.json").write_text(json.dumps({"source": "kolipakam2018dravlex",
    "language": "Gondi", "residualSourceRecords": len(records), "proposedRecords": len(accepted),
    "heldRecords": len(held), "unexaminedRecords": 0, "uniqueOwnership": True,
    "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(f"{len(accepted)} proposed records, {len(held)} holds, {sum(len(x.get('components', [x['parent']])) for x in accepted)} rows")
