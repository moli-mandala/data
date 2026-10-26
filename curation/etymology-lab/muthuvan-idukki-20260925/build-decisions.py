"""Conservative exact-record Muthuvan Idukki proposals; no accepted writes."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 768
item = {i: int(re.search(r"item (\d+)", r["Source"]).group(1))
        for i, r in enumerate(records, 1)}
proposals = {}


def add(number, parent, evidence, *, forms=None, starts=None, excludes=()):
    hits = []
    for i, rec in enumerate(records, 1):
        if item[i] != number or i in excludes:
            continue
        form = rec["Form"]
        if forms is not None and form not in forms:
            continue
        if starts is not None and not form.startswith(starts):
            continue
        assert i not in proposals, (i, form)
        proposals[i] = (parent, evidence)
        hits.append(i)
    assert hits, (number, parent)


# Every item is retained at its exact village/duplicate ID. Evidence refers to
# attested comparanda within the entire DEDR article, not its displayed head.
add(1, 5099, "DEDR 5099 directly gives Malayalam mēl and Kodagu me·lï ‘body’; Muthuvan melɨ is a close regional body form.", forms={"melɨ"})
add(6, 1448, "DEDR 1448 gives Tamil/Malayalam kātu and Iruḷa ka·du ‘ear’; Muthuvan kātɨ is a close local reflex.")
add(8, 5352, "DEDR 5352 gives Tamil vāy and Malayalam vāy ‘mouth’; Muthuvan bayɨ is the voiced local mouth form.")
add(13, 2023, "DEDR 2023 gives Tamil/Malayalam kai ‘hand, arm’; Muthuvan kɛyyɨ retains the hand/arm noun.")
add(24, 2760, "DEDR 2760 gives Malayalam ceṭi ‘dirty ground, dung’; the isolated Muthuvan ceḍi ‘feces’ retains that dung noun, with local stop voicing.", forms={"ceḍi"})
add(24, 4210, "DEDR 4210 directly gives Tamil/Malayalam pī ‘excrement’; the isolated Muthuvan pi response retains that noun.", forms={"pi"})
add(25, 1655, "DEDR 1655 gives Tamil kuṭi ‘house, settlement’ and Malayalam kuṭi ‘house, family’; the eight Muthuvan kuḍi village responses retain this settlement stem.")
add(26, 5393, "DEDR 5393 explicitly includes Tamil vīṭu ‘house, habitation’ and Malayalam vīṭu ‘house, freehold property’; Muthuvan vīḍɨ/vɨ̄ḍɨ/biḍɨ preserve that dwelling noun.")
add(46, 3045, "DEDR 3045 directly gives Tamil/Malayalam taṇṇīr/taṇṇi ‘water’; all eight Muthuvan responses preserve taṇṇi despite OCR spacing in the source.")
add(52, 1298, "DEDR 1298 gives Tamil kal and Malayalam kallu ‘stone’; Muthuvan kɛllɨ preserves the stone noun.")
add(57, 4481, "DEDR 4481 gives Tamil poṭi ‘powder, dust, ash’ and Malayalam poṭi ‘dust, powder’; the isolated Muthuvan poḍi responses fit ash/powder.", forms={"poḍi"})
add(90, 816, "DEDR 816 gives Tamil erumai and Malayalam eruma ‘buffalo’; Muthuvan ɛrumaḍɨ retains the buffalo base plus an animal suffix.", forms={"ɛrumaḍɨ"})
add(95, 3650, "DEDR 3650 gives Tamil nāy and Malayalam nāya ‘dog’; Muthuvan nāyɨ is a local dog form.")
add(112, 4198, "DEDR 4198 gives Tamil piḷḷai and Malayalam piḷḷa ‘child, daughter’; Muthuvan piḷḷa/puḷḷa variants are elicited as ‘daughter’.")
add(122, 3758, "DEDR 3758 gives Malayalam innalē and Iruḷa nä·ṯṯu ‘yesterday’; Muthuvan inaṭik/inaṭɛkɨ preserve the yesterday stem with a local ending.")
add(130, 3610, "DEDR 3610 gives Tamil nalla ‘good’; the isolated Muthuvan nallavan retains the good stem.", forms={"nallavan"})
add(148, 5496, "DEDR 5496 gives Tamil veḷḷa ‘white’ and Malayalam veḷuppu ‘whiteness’; Muthuvan vɛḷ-/bɛḷ- variants retain this white family.")
add(150, 1931, "DEDR 1931 includes Tamil cekkañ-ceku ‘deep red’ in the ce-/cev- red article; Muthuvan cɛkañŋɨ retains the cekkañ- red element.")
add(152, 474, "DEDR 474 gives Tamil iraṇṭu, Malayalam raṇṭu, and Kannada eraḍu ‘two’; Muthuvan raṇḍa/iraṇḍa/iraŋḍɨ retain the numeral.")
add(153, 5052, "DEDR 5052 gives Tamil mūṉṟu and Malayalam mūnnu ‘three’; Muthuvan mūntrɨ/mūntɨ are local three forms.")
add(157, 910, "DEDR 910 directly gives Kota e·y and Tamil ēr̤u ‘seven’; Muthuvan ɛ̄yɨ matches the Kota vowel-and-glide seven form.")
add(165, 5151, "DEDR 5151 includes Tamil ēvaṉ ‘who’ and other e-/y- animate interrogatives; Muthuvan ɛ̄van preserves this form.", forms={"ɛ̄van"})
add(166, 5151, "DEDR 5151 includes Tamil eṉṉa ‘what’ and related en-/e- interrogatives; Muthuvan ɛndan/ɛnd- variants retain that question base.")
add(167, 5151, "DEDR 5151 includes Telugu eṭa/ēḍa ‘where’ among e- place interrogatives; Muthuvan ɛduk/ɛdɨ/ɛ̄ḍa forms retain that base.")
add(168, 5151, "DEDR 5151 gives Tamil eppōtu ‘when’ and related epp- interrogatives; Muthuvan ɛppan/ɛppam/ɛppo retain the temporal stem.")
add(169, 5151, "DEDR 5151 gives Tamil ettaṉai ‘how many’; Muthuvan ɛttra/ɛtta ra are the quantity interrogative.")
add(171, 410, "DEDR 410 is the proximal i- demonstrative article, including Tamil itu/ivaṉ ‘this/this man’; Muthuvan itɨ and ivan retain that base.")
add(172, 1, "DEDR 1 is the distal a- demonstrative article, including Tamil atu/avaṉ ‘that/that man’; Muthuvan atɨ and avan retain that base.")
add(173, 410, "DEDR 410 contains the proximal itu/ivai ‘this/these’; four isolated Muthuvan itɨ responses reuse that proximal base for ‘these’.", forms={"itɨ"})
add(174, 1, "DEDR 1 contains the distal atu/avai ‘that/those’; four isolated Muthuvan atɨ responses reuse that distal base for ‘those’.", forms={"atɨ"})
add(183, 1124, "DEDR 1124 gives Tamil kaṭi and Malayalam kaṭikka ‘bite’; Muthuvan kaḍiś-/kaḍic- finite responses retain the bite stem.")
add(185, 1654, "DEDR 1654 gives Tamil kuṭi and Malayalam kuṭikka ‘drink’; Muthuvan kuḍiś- finite responses retain the drink stem.")
add(187, 707, "DEDR 707 gives Tamil uṟaṅku and Malayalam uṟaṅṅuka ‘sleep’; Muthuvan oraŋŋ-/uraŋŋ- finite responses preserve it.")
add(188, 1524, "DEDR 1524 gives Tamil kiṭa/kiṭanta and Malayalam kiṭakka ‘lie down’; Muthuvan kiḍ-/kɛḍ- responses preserve the lie-down stem.")
add(189, 480, "DEDR 480 gives Tamil iru and Malayalam irikkuka ‘sit’; Muthuvan irɨ-/irav- responses retain the sit stem.")
add(190, 3098, "DEDR 3098 directly gives Tamil taru past tant- and Malayalam taruka/tann- ‘give’; Muthuvan ta nd- finite responses retain the past stem despite source spacing.")
add(191, 811, "DEDR 811 gives Tamil eri and Malayalam eriyuka ‘burn’; Muthuvan ɛri-/ɛrun-/ɛran- finite responses retain this burn stem.", starts=("ɛri", "ɛrun", "ɛran"))
add(192, 2426, "DEDR 2426 gives Tamil cāku/cett- and Malayalam cāka/catt- ‘die’; Muthuvan tsatt-/tsak-/catt-/cak- responses retain this inflectional family.", starts=("tsatt", "tsak", "catt", "cak"), excludes={607})
add(193, 2132, "DEDR 2132 gives Tamil kol/koṉṟ- and Malayalam kolluka/koṇṇ- ‘kill’; Muthuvan koll-/kont-/kɔnt- forms preserve this kill family.")
add(195, 3582, "DEDR 3582 gives Tamil naṭa/naṭanta and Malayalam naṭakkuka ‘walk’; Muthuvan naḍa-/naḍann- finite forms retain the walk stem.")
add(196, 1041, "DEDR 1041 gives Tamil ōṭu/ōṭi and Malayalam ōṭuka ‘run’; the isolated Muthuvan ōḍiknān/ōḍitnɛ retain this verb.", forms={"ōḍiknān", "ōḍitnɛ"})
add(197, 4572, "DEDR 4572 gives Tamil pō/pōku and Malayalam pōka ‘go’; Muthuvan pokkaṇay/poyyan preserve this going stem.", forms={"pokkaṇay", "poyyan"})
add(199, 4430, "DEDR 4430 gives Tamil pēcu and Malayalam pēcuka ‘speak’; Muthuvan pesɨ-/peśɨ- finite responses retain this speech stem.", starts=("pes", "peś"))
add(200, 2017, "DEDR 2017 gives Tamil kēḷ/kēṭṭ- and Malayalam kēḷkkuka ‘hear, listen’; Muthuvan kɛṭ-/kēṭ-/kɛk- responses retain the hearing stem.")
add(201, 4091, "DEDR 4091 gives Tamil pār/pārtt- and Malayalam pārkka ‘see, look’; Muthuvan pār-/par-/patt- responses retain this looking stem.", starts=("pār", "par", "patt"))
add(201, 1443, "DEDR 1443 gives Tamil kāṇ/kaṇṭ- ‘see’; the two Muthuvan kaṇḍɛ̄y records preserve this second looking verb.", forms={"kaṇḍɛ̄y"})
add(202, 5160, "DEDR 5160 gives Tamil nāṉ, Kannada nān, and related ‘I’ forms; Muthuvan nānɨ/nan/nān are first-person singular.")
add(203, 3684, "DEDR 3684 gives Tamil/Malayalam nī ‘you’ singular; Muthuvan ni/nī retain the informal second-person singular.")
add(204, 3688, "DEDR 3688 gives Malayalam niṅṅaḷ and Kodagu niŋga ‘you’ plural/honorific; Muthuvan niŋŋa/niŋga variants are the formal response.")
add(205, 1, "DEDR 1 gives Tamil avaṉ ‘he’ in the distal demonstrative article; all eight Muthuvan avan responses are exact.")
add(206, 1, "DEDR 1 gives Tamil avaḷ ‘she’ in the distal demonstrative article; Muthuvan avaḷɨ/avaḷ/ava retain that feminine stem.")
add(207, 3647, "DEDR 3647 gives Tamil nām with oblique nam- ‘we’; all eight Muthuvan namma responses preserve the inclusive first-person plural stem.")
add(208, 5154, "DEDR 5154 directly gives Kodagu eŋga and Tamil eṅkaḷ oblique ‘we’ exclusive; Muthuvan ɛŋga/ɛŋŋa variants retain it.")
add(209, 3688, "DEDR 3688 gives Malayalam niṅṅaḷ and Kodagu niŋga ‘you’ plural; Muthuvan niŋŋa/niŋga preserve this form.")
add(210, 1, "DEDR 1 gives Tamil avar/avarkaḷ ‘they’ in the distal demonstrative family; Muthuvan avarɨ/avaŋga variants preserve that plural base.")

assert all(1 <= i <= len(records) for i in proposals)

source_conflicts = {1, 3, 17, 21, 23, 24, 27, 30, 41, 50, 58, 80, 97, 100,
                    102, 103, 111, 113, 114, 115, 116, 119, 120, 128, 130,
                    131, 132, 133, 134, 137, 142, 144, 145, 146, 150, 165,
                    170, 173, 174, 177, 178, 180, 181, 184, 186, 194, 196,
                    197, 199, 201}
loanlike = {1, 17, 21, 23, 30, 33, 35, 39, 40, 41, 42, 43, 60, 69, 77,
            78, 80, 81, 89, 117, 118, 120, 125, 126, 127, 141, 144, 145,
            175, 176}
compound_items = {15, 23, 27, 31, 32, 50, 56, 57, 80, 89, 90, 102, 103, 109,
                  113, 114, 115, 116, 119, 134, 138, 139, 144, 145, 161,
                  170, 173, 174, 175, 184, 186, 191, 194, 196, 199}

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
        reason = f"Appendix B3 item {n} records {form!r} ‘{gloss}’, but its elicitation, segment boundary, or cross-village sense needs source-level verification."
    elif n in loanlike:
        reason = f"{form!r} ‘{gloss}’ is loanlike or shared regional vocabulary; its immediate donor into this Muthuvan variety has not been attested."
    elif n in compound_items or " " in form:
        reason = f"{form!r} ‘{gloss}’ is a compound, finite clause, or OCR-spaced response whose full component analysis is not secure."
    else:
        reason = f"{form!r} ‘{gloss}’ lacks a secure dialect-specific comparison or immediate donor."
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
