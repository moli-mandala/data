"""Conservative Paniya Census Tamil Nadu research decisions; no graph writes."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 460
proposals = {}

def add(indices, number, evidence):
    for index in indices:
        assert index not in proposals
        proposals[index] = (number, evidence)

add([1], 1481, "DEDR 1481 gives Tamil kāṟṟu and Malayalam kāṟṟu 'air, wind'; the Census kāttu loses the rhotic cluster but retains the air sense.")
add([3], 2552, "DEDR 2552 lists Tamil iravu 'night' and iruḷ 'darkness', with Malayalam iravu/iruḷ; Census irəv is the night/darkness stem.")
add([6], 3266, "DEDR 3266 gives Tamil tī and Malayalam tī 'fire'; Census tivu has the same fire stem with a local ending.")
add([23], 3045, "DEDR 3045 directly lists Tamil/Malayalam taṇṇīr/taṇṇi 'water'; Census taṇṇi is the local short form.")
add([53], 4616, "DEDR 4616 gives Tamil makaṉ and Malayalam makan 'son'; Census magɛn is a voiced local son form.")
add([57], 4020, "DEDR 4020 lists Tamil paṟavai and Malayalam paṟava 'bird'; Census paṛəve retains the bird noun.")
add([58], 4355, "DEDR 4355 lists Tamil pūccai and Malayalam pūcca 'cat'; Census pūccɛ has the same stem and meaning.")
add([64], 2248, "DEDR 2248 gives Tamil/Malayalam kōr̤i and Kannada kōr̤i 'fowl, hen'; Census kōɻi is the corresponding hen noun.")
add([65], 3606, "DEDR 3606 lists Tamil nari 'jackal' and Kannada nari 'jackal, fox'; Census neri is a same-sense fox variant.")
add([73], 4307, "DEDR 4307 lists Tamil/Malayalam puli and Kannada puli 'tiger'; Census puli is exact.")
add([74], 5155, "DEDR 5155 lists Malayalam āma 'tortoise' and Tamil āmai; Census āmə preserves the local tortoise noun.")
add([105], 2897, "DEDR 2897 directly lists Tamil/Malayalam cōṟu 'boiled rice, food'; Census sōṛu fits the cooked-rice portion of the source's combined rice prompt.")
add([126], 3408, "DEDR 3408 has Malayalam tēṅṅa and Kodagu teŋge 'coconut'; Census tēŋgɛ closely matches the latter regional form.")
add([155], 1249, "DEDR 1249 lists Tamil kaippu and Malayalam kaippu 'bitterness'; Census kaippɛ is the same bitter stem.")
add([195], 486, "DEDR 486 directly gives Tamil/Malayalam irumpu 'iron'; Census irumbu is the voiced local counterpart.")
add([198], 5496, "DEDR 5496(a) lists Tamil veḷḷi 'silver' and related white/bright forms; Census boḷḷi has the regional b-initial silver shape.")
add([202], 2826, "DEDR 2826 gives Tamil añcu and Malayalam añcu 'five'; Census añźu preserves the five numeral with affricate voicing.")
add([205], 741, "DEDR 741 has Tamil ūtu and Malayalam ūtuka 'to blow'; Census ūtiñcai adds the elicited Paniya verb ending.")
add([208], 3376, "DEDR 3376(a) lists Tamil tūkku 'lift, hoist' and Malayalam tūkkuka 'take up'; Census dūkkuñcai is a local carry/lift verb within the source's broad carrying prompt.")
add([219], 3570, "DEDR 3570 gives Tamil nakku and Malayalam nakkuka 'lick'; Census nakkiñcai retains the lick stem with verbal ending.")
add([221], 1299, "DEDR 1299 lists Tamil kala 'mix' and Malayalam kalaruka 'be mixed, mingle'; Census kalākiñcai is a local mix verb, distinct from the confusion article 1303.")
add([222], 3259, "DEDR 3259 lists Tamil tiṟa 'open a door/eyes' and Malayalam tuṟakka; Census duṛākiñcai keeps the open stem.")
add([223], 4027, "DEDR 4027 lists Tamil paṟi and Malayalam paṟikka 'pluck, pick'; Census paṛikiñcai is the matching pick verb.")
add([227], 5549, "DEDR 5549 has Tamil vai and Malayalam vekka 'put, place'; Census vekkiñcai is a local put verb with regular elicited ending.")
add([229], 5421, "DEDR 5421 lists Tamil vil and Malayalam vilkka 'sell'; Census viḷḷikkiñcai retains the sell stem with local gemination and the elicited ending.")
add([230], 5034, "DEDR 5034 lists Tamil mūṭu 'cover, shut' and Malayalam mūṭuka; Census mūṭuñcai is the shut verb.")
add([232], 3435, "DEDR 3435 gives Tamil teḷi and Malayalam teḷikka 'sprinkle, scatter'; Census teḷikiñcai matches this action.")
add([234], 3323, "DEDR 3323 lists Tamil tuppu and Malayalam tuppuka 'spit'; Census tuppiñcai has the same stem.")
add([237], 859, "DEDR 859 gives Tamil eṟi and Malayalam eṟiyuka 'throw, fling'; Census eṛiñcai is a local throw form.")
add([239], 3301, "DEDR 3301 gives Tamil tuṭai and Malayalam tuṭekka 'wipe'; Census duḍaikiñcai retains the wipe stem with medial voicing.")
add([241], 1624, "DEDR 1624 lists Tamil kīṟu 'slit, cut' and Malayalam kīṟuka 'tear, rend'; Census gīruñcai is a cut verb with voiced initial.")
add([246], 5008, "DEDR 5008 specifically lists Tamil muṟi 'break as a stick' and Malayalam muṟikka 'break'; Census muṛikiñcai is the split-wood stem.")
add([248], 2132, "DEDR 2132 lists Tamil kol and Malayalam kolluka 'kill'; Census kolluñcai preserves this verb.")
add([252], 4065, "DEDR 4065 lists Tamil pāṭu and Malayalam pāṭuka 'sing'; Census pāḍiñcai has the expected voiced stop.")
add([269], 4087, "DEDR 4087 gives Tamil pāy 'run, leap' and Malayalam pāyuka; Census pāyiñcai matches the run sense.")
add([277], 3582, "DEDR 3582 lists Tamil naṭa and Malayalam naṭakkuka 'walk'; Census naḍākiñcāy preserves the walk stem.")
add([280], 5093, "DEDR 5093 lists Tamil mēy and Malayalam mēyuka 'graze'; Census mēyikiñcai retains the graze stem.")
add([281, 285], 2119, "DEDR 2119 gives Tamil koy 'cut, reap' and Malayalam koyka 'reap'; both Census koyiñcai records have harvest/reap senses.")
add([289], 1124, "DEDR 1124 lists Tamil kaṭi and Malayalam kaṭikka 'bite'; Census kaḍikkiñcai is a voiced local bite form.")
add([291], 1654, "DEDR 1654 lists Tamil kuṭi and Malayalam kuṭikka 'drink'; Census kuḍikiñcai retains the drinking stem.")
add([292], 3263, "DEDR 3263 has Tamil tiṉ and Malayalam tiṉnuka 'eat'; Census tiñcai is a shortened eat verb.")
add([310], 1416, "DEDR 1416 lists Tamil kā 'guard, wait' and Malayalam kākkuka 'guard, wait'; Census kātukkuñcai preserves the wait/guard stem with verbal ending.")
add([317], 793, "DEDR 793 lists Tamil eṇṇu and Malayalam eṇṇuka 'count, reckon'; Census eṇṇiñcai matches the counting stem.")
add([318], 314, "DEDR 314 gives Tamil aṟi and Malayalam aṟiyuka 'know, understand'; Census aṛiñcai is the corresponding know verb.")
add([336], 1832, "DEDR 1832 lists Tamil kuḷi and Malayalam kuḷikka 'bathe'; Census kuḷikiñcai is the bathe record, separate from the identical form misglossed 'buy'.")
add([342], 1443, "DEDR 1443 directly lists Tamil kāṭṭu and Malayalam kāṭṭuka 'show'; Census kāṭṭiñcai retains that causative see/show stem.")
add([387, 405], 3584, "DEDR 3584 gives Tamil naṭuvu and Kannada naḍuve 'middle, between'; both Census naḍuvili records fit the centre/between adverb.")
add([403], 4205, "DEDR 4205 gives Tamil/Malayalam piṉṉē 'after, afterwards'; Census pinnɛ retains that temporal base.")
add([417], 5160, "DEDR 5160 gives Tamil nāṉ and Kannada nān 'I'; Census nānu is the same first-person pronoun with final vowel.")
add([422, 423], 5160, "DEDR 5160 gives first-person oblique en- alongside the nominative 'I'; Census enākke 'to me' and ennā 'my' retain that oblique base.")
add([424, 436], 3647, "DEDR 3647 has Tamil/Malayalam nām 'we' and Kannada nāvu 'we'; Census nām 'our' and nāmu 'we' share the inclusive first-person plural stem.")
add([437, 438], 5151, "DEDR 5151 has the interrogative e- family, including Tamil eṉṉa 'what' and eppōtu 'when'; Census enna/eppā preserve the respective question stems.")
add([440, 442], 5151, "DEDR 5151 lists Tamil ār/yār 'who' and related case forms; Census āru 'who' and ārne 'whom' preserve the animate interrogative.")
add([443], 3688, "DEDR 3688 lists Kannada nīvu and Tamil nīvir 'you plural'; Census niyvu is a close plural/polite second-person form, although the source gloss does not mark number.")

assert set(proposals) <= set(range(1, len(records) + 1))

# These are source-level form/gloss collisions, not corrections to the source.
mismatched = {13, 14, 17, 19, 20, 22, 34, 35, 36, 40, 41, 43, 45, 46, 47, 48,
              59, 60, 67, 70, 71, 72, 75, 79, 80, 82, 83, 84, 85, 86, 88, 89,
              90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 103, 104, 114,
              115, 116, 117, 118, 119, 121, 123, 128, 129, 131, 132, 133, 135,
              136, 137, 138, 140, 141, 143, 147, 157, 158, 161, 162, 164, 166,
              168, 171, 173, 174, 201, 203, 212, 235, 264, 266, 267, 279, 283,
              286, 293, 294, 305, 320, 321, 324, 325, 326, 327, 328, 329,
              330, 331, 332, 333, 337, 341, 347, 355, 357, 358, 359, 361, 362,
              366, 368, 369, 372, 376, 377, 378, 379, 380, 381, 385, 394, 395,
              397, 407, 414, 415, 418, 419, 420, 427, 430, 431, 432, 433, 434,
              441, 445, 446, 451, 452, 453, 454, 456, 457, 458, 460}
assert not mismatched & proposals.keys()
loanlike = {4, 5, 15, 21, 24, 35, 38, 52, 106, 109, 110, 113, 120, 124, 139,
            149, 154, 159, 175, 176, 178, 179, 182, 183, 185, 188, 189, 193,
            194, 196, 197, 199, 200, 201, 363, 370, 373, 378, 385, 398, 400, 446}

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
    if index in mismatched:
        reason = f"The Census form {form!r} is inconsistent with its recorded gloss ‘{gloss}’ or collides with a different same-source sense; preserve the source row pending source-level verification."
    elif form.lower() in {"loom", "no data", "no vord"}:
        reason = "The response appears to be an English placeholder or has no usable Paniya lexical form."
    elif any(c in form for c in [" ", "[", "]"]):
        reason = f"The Census response {form!r} ‘{gloss}’ is multiword or OCR-spaced; its components and source alignment are not secure enough for a whole-form etymology."
    elif index in loanlike:
        reason = f"{form!r} ‘{gloss}’ is loanlike or shared regional vocabulary, but its immediate donor in Paniya has not been attested."
    else:
        reason = f"{form!r} ‘{gloss}’ lacks a secure dialect-specific comparison, donor, or stem analysis; a form/gloss resemblance alone is insufficient."
    held.append({"record": rec, "reason": reason, "status": "research-hold", "surveyIndex": index})

parent_ids = {row["parent"] for row in accepted}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as stream:
    for row in csv.DictReader(stream):
        if row["Entry_ID"] in parent_ids:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parent_ids, sorted(parent_ids - set(articles))
assert len(accepted) + len(held) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
(P / "source-gloss-conflict-indices.json").write_text(json.dumps(sorted(mismatched), indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(len(accepted), "proposed records,", len(held), "holds,", len(articles), "parent articles")
