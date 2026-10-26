"""Mannan Idukki exact-record etymology proposals, pending review only."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 679
item = {i: int(re.search(r"item (\d+)", r["Source"]).group(1))
        for i, r in enumerate(records, 1)}
proposals = {}


def add(number, parent, evidence, *, forms=None, starts=None, excludes=()):
    hits = []
    for i, record in enumerate(records, 1):
        if item[i] != number or i in excludes:
            continue
        form = record["Form"]
        if forms is not None and form not in forms:
            continue
        if starts is not None and not form.startswith(starts):
            continue
        assert i not in proposals, (i, form)
        proposals[i] = (parent, evidence)
        hits.append(i)
    assert hits, (number, parent)


# Comparanda are read from the whole cached DEDR articles. The source repeats
# elicitation prompts across villages; every response keeps its own exact ID.
add(1, 5099, "DEDR 5099 directly gives Malayalam mēl and Kodagu me·lï ‘body’; the single Mannan melɨ response retains this regional body form.", forms={"melɨ"})
add(16, 5409, "DEDR 5409 gives Tamil viral, Malayalam viral, and Kannada beral ‘finger’; Mannan biraḷ shares the regional b-initial finger form.")
add(23, 644, "DEDR 644 directly gives Malayalam moḷḷu ‘urine’, Iruḷa maḷḷu, and Kota moḷ; Mannan mɔḷḷɨ/muḷḷɨ are the same urine noun.")
add(24, 4210, "DEDR 4210 directly gives Tamil/Malayalam pī ‘excrement’ and Telugu pīyi; Mannan pi/piyɨ are same-sense forms.")
add(25, 1655, "DEDR 1655 gives Tamil/Malayalam kuṭi ‘house, settlement, lineage’; the isolated Mannan kuḍi is the village/settlement term.", forms={"kuḍi"})
add(26, 1904, "DEDR 1904 gives Malayalam kūra ‘hut, thatch’ and Tamil kūrai ‘roof, small hut’; Mannan kūra/kurɨ are the dwelling word.")
add(46, 3045, "DEDR 3045 directly gives Tamil/Malayalam taṇṇīr/taṇṇi ‘water’; all seven Mannan responses visibly contain taṇṇi despite OCR spacing, retained in the original source strings.")
add(55, 3266, "DEDR 3266 gives Tamil/Malayalam tī ‘fire’; Mannan tīyɨ has the fire stem.")
add(59, 4666, "DEDR 4666 gives Tamil maṇ ‘earth, dust’ and Malayalam maṇṇu ‘earth, soil’; Mannan maṇṇɨ is the dust/earth word.", forms={"maṇṇɨ"})
add(122, 3758, "DEDR 3758 gives Kannada ninne and Kodagu ninna·ndï ‘yesterday’; Mannan ninaṭ preserves the nin- yesterday stem.")
add(133, 5222, "DEDR 5222 gives Tamil vaṭi ‘dry up’ and Malayalam vaṭiyuka ‘dry up’; Mannan vaḍiñ-/baḍiñ- responses retain this dry stem.", starts=("vaḍiñ", "baḍiñ"))
add(133, 601, "DEDR 601 gives Tamil uṇaṅku and Malayalam uṇaṅṅuka ‘dry’; Mannan uṇand-/uṇante variants preserve the dry stem.", starts="uṇan")
add(135, 1594, "DEDR 1594 lists Tamil ciṟiya ‘small’ and ciṟu ‘small, short’; Mannan curiya/curiyatɨ are the short forms.")
add(148, 5496, "DEDR 5496 gives Tamil veḷḷa ‘white’ and related Malayalam/Kannada white forms; Mannan vaḷḷa/veḷḷa/baḷḷa variants retain that stem.")
add(152, 474, "DEDR 474 gives Tamil iraṇṭu, Malayalam raṇṭu, and Kannada eraḍu ‘two’; Mannan iraŋḍɨ is the numeral.")
add(165, 5151, "DEDR 5151 lists Tamil ār/yār and related Dravidian ‘who’ forms; Mannan ārɨ preserves the animate interrogative.")
add(166, 5151, "DEDR 5151 includes Tamil eṉṉa ‘what’ and other e-/en- interrogatives; Mannan endan-/ɛndan-/ɛnd- responses retain that base.")
add(167, 5151, "DEDR 5151 includes Telugu eṭa/ēḍa ‘where’ and related e- place interrogatives; Mannan ɛ̄ḍa/ɛ̄ḍekku fits this set.")
add(168, 5151, "DEDR 5151 gives Tamil eppōtu ‘when’ and related epp- interrogatives; Mannan ɛppa/ɛppɨ/yappɨ are local temporal forms.")
add(169, 5151, "DEDR 5151 gives Tamil ettaṉai ‘how many’; Mannan ɛttina/ɛtta na/ettra preserves the quantity interrogative.")
add(171, 410, "DEDR 410 is the proximal i- demonstrative article, including Tamil itu ‘this’; all seven Mannan itɨ responses retain that form.")
add(172, 1, "DEDR 1 is the distal a- demonstrative article, including Tamil atu ‘that’; Mannan atɨ retains that form.", forms={"atɨ"})
add(173, 410, "DEDR 410 contains Tamil proximal itu/ivai ‘this/these’; the one isolated Mannan itɨ is a proximal response to ‘these’.", forms={"itɨ"})
add(174, 1, "DEDR 1 contains Tamil distal atu/avai ‘that/those’; the isolated Mannan atɨ is a distal response to ‘those’.", forms={"atɨ"})
add(183, 1124, "DEDR 1124 gives Tamil kaṭi and Malayalam kaṭikka ‘bite’; Mannan kaḍic- responses retain the bite stem and finite morphology.")
add(187, 707, "DEDR 707 gives Tamil uṟaṅku and Malayalam uṟaṅṅuka ‘sleep’; the Mannan uraŋganɨ response retains the sleep verb.")
add(188, 1524, "DEDR 1524 gives Tamil kiṭa/kiṭanta and Malayalam kiṭakka ‘lie down’; Mannan kiḍand-/kiḍak- variants are the same verb.")
add(190, 3098, "DEDR 3098 directly gives Tamil taru past tant- and Malayalam taruka/tann- ‘give’; Mannan ta nd- responses retain the past stem despite source spacing.")
add(191, 1458, "DEDR 1458 gives Tamil kāntu and Malayalam kāntuka ‘burn, be hot’; Mannan kānd-/kāndi- responses preserve this burn verb.", starts="kānd")
add(192, 2426, "DEDR 2426 gives Tamil cāku/cett- and Malayalam cāka/catt- ‘die’; Mannan cak-/catt- forms answer both finite and negative death clauses.")
add(193, 2132, "DEDR 2132 gives Tamil kol/koṉṟ- and Malayalam kolluka/koṇṇ- ‘kill’; Mannan koll-/koṇṇ-/konn- forms preserve that inflectional family.")
add(194, 4020, "DEDR 4020 gives Tamil paṟa/paṟanta and Malayalam paṟakka ‘fly’; Mannan pariya-/paranda- responses retain the flying stem.", starts=("pariyac", "pariyañj", "parand"))
add(195, 3582, "DEDR 3582 gives Tamil naṭa/naṭanta and Malayalam naṭakkuka ‘walk’; Mannan naḍan-/naḍand- forms retain that verb.", starts=("naḍan", "naḍand"))
add(196, 1041, "DEDR 1041 gives Tamil ōṭu/ōṭi and Malayalam ōṭuka ‘run’; Mannan ōḍi- responses retain this verb.")
add(199, 2855, "DEDR 2855 gives Tamil col/colli and Malayalam colluka ‘speak, say’; Mannan colliy- forms retain that stem.")
add(200, 2017, "DEDR 2017 gives Tamil kēḷ/kēṭṭ- and Malayalam kēḷkkuka ‘hear, listen’; Mannan kēṭṭ-/kēṭ-/keṭṭ- responses retain the heard form.", starts=("kēṭ", "keṭṭ"))
add(201, 3794, "DEDR 3794 gives Tamil nōkku/nōkki and Malayalam nōkkuka ‘look’; Mannan nōkk-/nokk-/ṇokk- variants retain this verb.", starts=("nōkk", "nokk", "ṇokk"))
add(201, 1443, "DEDR 1443 gives Tamil kāṇ/kaṇṭ- ‘see, look’; the Mannan kāṇ and kanḍēy responses preserve this other seeing stem.", forms={"kāṇ", "kanḍēy"})
add(202, 5160, "DEDR 5160 gives Tamil nāṉ, Malayalam ñān, and Kannada ān/nān ‘I’; Mannan ɛ̄n is the first-person singular.")
add(203, 3684, "DEDR 3684 gives Tamil/Malayalam nī ‘you’ singular; Mannan ni is the informal second-person singular.")
add(204, 3684, "DEDR 3684 includes Tamil nī/nīr singular and honorific ‘you’; Mannan ni is retained as the formal singular response.", forms={"ni"})
add(204, 3688, "DEDR 3688 gives Malayalam niṅṅaḷ and Kodagu niŋga ‘you’ plural/honorific; Mannan niŋŋa/niŋgɛ share this respectful/plural stem.", starts="niŋ")
add(205, 1, "DEDR 1 gives Tamil avaṉ ‘he’ in the distal demonstrative article; Mannan avan is exact.")
add(206, 1, "DEDR 1 gives Tamil avaḷ ‘she’ in the distal demonstrative article; Mannan ava/avaḷ retain that feminine form.")
add(207, 5154, "DEDR 5154 directly gives Kodagu naŋga ‘we’ alongside eŋga; Mannan naŋgɨ/naŋga is the inclusive first-person plural.")
add(208, 5154, "DEDR 5154 directly gives Kodagu eŋga ‘we’ and Tamil eṅkaḷ oblique, marked exclusive; Mannan ɛŋgɛ/ɛŋga forms retain the exclusive stem.")
add(209, 3688, "DEDR 3688 gives Malayalam niṅṅaḷ and Kodagu niŋga ‘you’ plural; all seven Mannan niŋgɛ responses retain that plural form.")
add(210, 1, "DEDR 1 gives Tamil avar/avarkaḷ ‘they’ in the distal demonstrative family; Mannan avarɨ/avara are plural responses.")

assert all(1 <= i <= len(records) for i in proposals)

source_conflicts = {1, 4, 17, 21, 23, 25, 27, 31, 33, 34, 41, 49, 50, 80,
                    89, 90, 98, 100, 109, 120, 123, 130, 131, 133, 137,
                    142, 144, 145, 170, 172, 173, 174, 180, 181, 182, 184,
                    186, 197, 200, 204}
loanlike = {3, 4, 17, 35, 38, 40, 41, 53, 57, 60, 69, 78, 80, 81,
            125, 126, 127, 130, 131, 176, 180}
multiword_items = {15, 27, 31, 32, 49, 50, 71, 80, 89, 110, 138, 139,
                   144, 145, 161, 173, 174, 175, 176, 179, 184, 186, 194, 195}

accepted = []
for i, (number, evidence) in sorted(proposals.items()):
    rec = records[i - 1]
    accepted.append({"record": rec, "parent": f"d{number}", "kind": "reflex",
                     "citation": rec["Source"] + f";dedr[entry {number}]",
                     "evidence": evidence, "status": "pending-review", "surveyIndex": i})

held = []
for i, rec in enumerate(records, 1):
    if i in proposals:
        continue
    n, form, gloss = item[i], rec["Form"], rec["Gloss"]
    if n in source_conflicts:
        reason = f"Appendix B3 item {n} records {form!r} ‘{gloss}’, but its elicitation, segment boundary, or cross-village sense is unclear; preserve this exact response pending source-level verification."
    elif n in loanlike:
        reason = f"{form!r} ‘{gloss}’ is loanlike or shared regional vocabulary; its immediate donor into this Mannan variety is not attested."
    elif n in multiword_items or " " in form:
        reason = f"{form!r} ‘{gloss}’ is a compound, finite clause, or OCR-spaced response whose full component analysis is not secure."
    else:
        reason = f"{form!r} ‘{gloss}’ lacks a secure dialect-specific comparison or immediate donor; a broad semantic resemblance is insufficient."
    held.append({"record": rec, "reason": reason, "status": "research-hold", "surveyIndex": i})

parents = {x["parent"] for x in accepted}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as stream:
    for row in csv.DictReader(stream):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, sorted(parents - set(articles))
assert len(accepted) + len(held) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(len(accepted), "proposed records,", len(held), "holds,", len(articles), "parent articles")
