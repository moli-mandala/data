"""Conservative Urali Census research decisions; no graph or sidecar writes."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 441
proposals = {}


def add(indices, number, comparison):
    for index in indices:
        assert index not in proposals, index
        rec = records[index - 1]
        proposals[index] = (number, f"DEDR {number} records {comparison}; Census Urali {rec['Form']} ‘{rec['Gloss']}’ retains that stem and sense.")


add([8], 1682, "Kannada guḍḍa/guḍḍu ‘hill’ and guḍḍe ‘heap’, with Tulu guḍḍè ‘hill’")
add([21], 3690, "Tamil nīr and Kannada nīru ‘water’")
add([25], 4198, "Tamil piḷḷai and Malayalam piḷḷa ‘child’")
add([29], 131, "Tamil aṇṇaṉ and Malayalam aṇṇan ‘elder brother’; the Census prompt conflates older and younger, but anna is the elder term")
add([30], 3085, "Tamil and Malayalam tampi ‘younger brother’; the Census prompt conflates older and younger, but tambi is the younger term")
add([32], 399, "Tamil and Malayalam āṇ ‘male’")
add([35], 4813, "Tamil māmā and Malayalam māman ‘maternal uncle’")
add([46], 23, "Tamil akkā/akkai and Kannada akka ‘elder sister’; the Census prompt conflates older and younger")
add([49], 4395, "Tamil peṇṭu ‘woman, wife’ and related wife forms")
add([51], 4355, "Tamil pūṉai and Kodagu pu·ñe ‘cat’")
add([52, 57], 2248, "Tamil and Malayalam kōr̤i and Kota ko·y ‘fowl’; the two Census koyi rows separately elicit cock and hen")
add([56], 4885, "Tamil, Malayalam, and Kannada mīn ‘fish’")
add([59], 5152, "Kota a·ṛ ‘goat’ beside Tamil āṭu and Malayalam āṭu")
add([61], 2115, "Tamil and Malayalam kompu ‘horn’")
add([62], 1711, "Kannada kudire/kudure and Kota kudyr ‘horse’")
add([66], 4307, "Tamil, Malayalam, and Kannada puli ‘tiger’")
add([67], 5155, "Kota e·m(b) and Kannada ēme ‘tortoise’")
add([70], 5488, "Kannada bennu ‘back’")
add([72, 77], 5409, "Tamil/Malayalam viral and Kannada beral ‘finger’; the duplicate Census records are preserved")
add([74], 1448, "Tamil/Malayalam kātu and Iruḷa ka·du ‘ear’")
add([82], 1479, "Tamil/Malayalam kāl ‘leg, foot’")
add([87], 5024, "Tamil/Malayalam mūkku ‘nose’")
add([89], 3633, "Kannada nālage/nālige ‘tongue’ beside Kota na·lg")
add([90], 3986, "Tamil pal and Malayalam pallu ‘tooth’")
add([94], 4096, "Tamil/Malayalam/Kannada pāl ‘milk’")
add([95], 2674, "Tamil/Malayalam/Kannada uppu ‘salt’")
add([96], 4240, "Kannada poge and Tamil pukai ‘smoke’")
add([97], 215, "Tamil arici ‘husked rice’; the Census prompt also mentions cooked rice, which is not claimed for this form")
add([112], 5028, "Iruḷa mu·ŋga and Tamil mūṅkil ‘bamboo’")
add([113], 382, "Tamil ālamaram ‘banyan tree’ within the article's āl tree family")
add([116], 3408, "Tamil tēṅkāy and Kota ten ka·y ‘coconut’")
add([121], 2673, "Kannada soppu ‘green leaves’")
add([124], 3753, "Tamil nellu and Malayalam nellu ‘paddy’")
add([126], 5535, "Tamil/Malayalam vēr ‘root’")
add([127], 530, "Tamil iṉippu ‘sweetness’")
add([140], 1249, "Tamil/Malayalam kaippu ‘bitterness’")
add([151], 5496, "Tamil veḷ/veṇ and Kannada beḷḷa ‘white’")
add([153], 3825, "Tamil paci ‘hunger, to be hungry’")
add([155], 3793, "Tamil nōvu, Malayalam nōvu, and Kannada nōvu ‘pain, ache’")
add([176], 1278, "Kannada kabbiṇa and Toda kabïn ‘iron’ amid black/iron terms")
add([179], 5496, "Tamil veḷḷi ‘silver’ within the white/bright article")
add([183], 3655, "Tamil nāl/nālu ‘four’")
add([184], 2826, "Tamil/Malayalam añcu ‘five’")
add([185], 2485, "Tamil āṟu and Malayalam āṟu ‘six’")
add([186], 910, "Tamil/Malayalam ēr̤u ‘seven’")
add([187], 784, "Tamil/Malayalam eṭṭu ‘eight’")
add([188], 1025, "Tamil/Malayalam ompatu ‘nine’")
add([189], 3918, "Tamil pattu ‘ten’")
add([191], 1147, "Tamil kaṭṭu ‘tie, fasten, bind’")
add([198, 205], 4797, "Kannada māḍu ‘do, make’ and Malayalam māṭuka ‘build’; the repeated Census mādu responses remain separate")
add([203], 4148, "Tamil piṭi and Kannada hiḍi ‘hold, grasp’")
add([204], 3570, "Tamil/Malayalam nakku ‘lick’")
add([206], 1299, "Tamil kala ‘mix’ and Malayalam kalaruka ‘mingle’")
add([207], 3259, "Tamil tiṟa ‘open a door or eyes’")
add([216], 4834, "Tamil māṟu ‘sell, exchange’")
add([218], 4915, "Tamil muccu and Kota muc- ‘cover, shut’")
add([222], 3323, "Tamil/Malayalam tuppu ‘spit’")
add([225, 226], 859, "Tamil eṟi ‘throw’; both eri and yeri are separately recorded")
add([230], 5478, "Tamil/Malayalam veṭṭu ‘cut’")
add([232], 3549, "Tamil tōṇṭu ‘dig’")
add([238], 2132, "Tamil kol and Malayalam kolluka ‘kill’")
add([241], 77, "Tamil aṭi ‘beat, strike’, including the source's drum beating")
add([242], 4065, "Tamil pāṭu and Malayalam pāṭuka ‘sing’")
add([243], 5270, "Tamil varu ‘come’ and its arrival sense")
add([246], 916, "Tamil ēṟu ‘climb, ascend’")
add([247], 5270, "Tamil vā, imperative of varu ‘come’")
add([253], 4020, "Tamil paṟa ‘fly’")
add([256], 2053, "Tamil koṭu ‘give’")
add([257], 4572, "Tamil pō/ pōku ‘go’")
add([268], 3582, "Tamil naṭa and Malayalam naṭakkuka ‘walk’")
add([271], 5093, "Tamil/Malayalam mēy ‘graze’")
add([278], 5401, "Tamil vittu and Kannada bittu ‘sow seed’")
add([280], 1124, "Tamil kaṭi and Malayalam kaṭikka ‘bite’")
add([282], 1654, "Tamil kuṭi and Malayalam kuṭikka ‘drink’")
add([290], 2855, "Tamil col and Malayalam col(l)uka ‘say’")
add([295], 2426, "Tamil cāku and Malayalam cāka ‘die’")
add([296, 298], 4760, "Tamil maṟai ‘hide, conceal’; the duplicate Census responses remain separate")
add([300], 1416, "Tamil kā ‘guard, wait’ and Malayalam kākkuka ‘wait’")
add([313], 55, "Tamil añcu and Malayalam añcuka ‘fear’")
add([315], 3569, "Tamil naku ‘laugh, smile’")
add([317], 707, "Iruḷa roṅgu ‘sleep’, explicitly attested within the article")
add([320], 3480, "Tamil toṭu ‘touch’")
add([323], 4960, "Tamil muttu and Malayalam muttuka ‘kiss’")
add([327], 5336, "Tamil vāṅku ‘buy, receive’")
add([332], 1443, "Tamil kāṭṭu ‘show’ within the kāṇ ‘see’ article")
add([344], 621, "Iruḷa uddya and Kannada udda ‘long, length’")
add([372, 389], 3584, "Kannada naḍuve and Tamil naṭuvu ‘middle, between’")
add([388, 411], 4205, "Tamil piṉpu/piṉṉē ‘after, afterwards’")
add([396], 410, "Tamil iṅku/iṅkē ‘here’ in the proximal demonstrative article")
add([399], 5160, "Kannada nān and Tamil nāṉ ‘I’")
add([403], 4597, "Tamil pōla ‘like, resembling’")
add([404, 405], 5160, "Kannada oblique nan- ‘my, me’ beside Tamil nāṉ ‘I’")
add([406, 418], 3647, "Tamil/Malayalam nām and Kannada oblique nam- ‘we, our’")
add([407], 2559, "Tamil illai ‘not, non-existent’")
add([409], 1, "Tamil a- distal demonstrative ‘that’")
add([412], 1, "Tamil avai ‘those’ in the distal demonstrative article")
add([413], 1, "Tamil aṅku ‘there’ in the distal demonstrative article")
add([414, 415], 1, "Tamil avar/avarkaḷ ‘they, their’ in the distal demonstrative article")
add([419, 420, 421], 5151, "Tamil eṉṉa ‘what’, eppōtu ‘when’, and eṅku ‘where’ in the interrogative article")
add([422, 424], 5151, "Tamil ār/yār ‘who’ with case forms in the interrogative article")
add([425, 426], 3684, "Tamil/ Kannada second-person oblique nin- ‘you, your’")
add([432], 240, "Tamil alai ‘wave, ripple’ and Malayalam ala ‘wave’")
add([437], 2661, "Tamil cuṇṭeli and Kannada suṇḍil-ili ‘mouse, shrew’")
add([439], 4879, "Kannada mīse and Tamil mīcai ‘moustache’")
add([440], 4460, "Tamil pokkuḷ and Kannada pokkur̤ ‘navel’")

# These specific Census form/gloss combinations require source-level checking.
conflicts = {12, 14, 17, 36, 38, 80, 83, 104, 105, 125, 162, 173, 193, 195,
             214, 231, 235, 259, 283, 284, 292, 301, 303, 316, 326, 339,
             352, 359, 400, 401, 402, 416, 438, 441}
assert not conflicts & proposals.keys()
loanlike = {2, 4, 5, 15, 34, 53, 63, 91, 98, 100, 101, 106, 109, 114,
            115, 119, 120, 122, 129, 130, 136, 139, 143, 145, 149, 152,
            157, 158, 159, 161, 163, 168, 169, 170, 171, 174, 175, 177,
            178, 181, 182, 185, 186, 187, 188, 189, 342, 348, 353, 355,
            363, 370, 373, 376, 378, 385, 427, 436}
# Ordinary numeral forms are linked above even when shared with neighboring languages.
loanlike -= set(proposals)

accepted = []
for index, (number, evidence) in sorted(proposals.items()):
    rec = records[index - 1]
    accepted.append({"record": rec, "parent": f"d{number}", "kind": "reflex",
                     "citation": rec["Source"] + f";dedr[entry {number}]",
                     "evidence": evidence, "status": "pending-review", "surveyIndex": index})

held = []
for index, rec in enumerate(records, 1):
    if index in proposals:
        continue
    form, gloss = rec["Form"], rec["Gloss"]
    if index in conflicts:
        reason = f"Census Urali {form!r} ‘{gloss}’ has a form/gloss conflict, duplicate with a different elicitation, or unclear transcription; source-level verification is needed."
    elif any(c in form for c in [" ", "[", "]"]):
        reason = f"The Census response {form!r} ‘{gloss}’ is multiword or OCR-spaced; the whole-form etymology and component order are not secure."
    elif index in loanlike:
        reason = f"{form!r} ‘{gloss}’ is loanlike or widely shared regional vocabulary, but its immediate donor into Urali is not attested."
    else:
        reason = f"{form!r} ‘{gloss}’ lacks a secure dialect-specific comparison, immediate donor, or stem analysis."
    held.append({"record": rec, "reason": reason, "status": "research-hold", "surveyIndex": index})

parents = {item["parent"] for item in accepted}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as stream:
    for row in csv.DictReader(stream):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {key: row[key] for key in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, sorted(parents - set(articles))
assert len(accepted) + len(held) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
(P / "source-gloss-conflict-indices.json").write_text(json.dumps(sorted(conflicts), indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(len(accepted), "proposed records,", len(held), "holds,", len(articles), "parent articles")
