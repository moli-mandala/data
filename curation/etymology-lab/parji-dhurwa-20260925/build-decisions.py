"""Conservative Parji Dhurwa proposals by exact source item and list record."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 495
item = {i: int(re.search(r"item (\d+)", r["Source"]).group(1))
        for i, r in enumerate(records, 1)}
proposals = {}


def add(number, parent, evidence, *, forms=None, excludes=()):
    hits = []
    for i, rec in enumerate(records, 1):
        if item[i] != number or i in excludes or (forms is not None and rec["Form"] not in forms):
            continue
        assert i not in proposals, (i, rec["Form"])
        proposals[i] = (parent, evidence)
        hits.append(i)
    assert hits, (number, parent)


# The DEDR identifier names a whole article. Each note names the relevant
# attested Parji segment or a diagnostic cross-language comparison within it.
add(10, 4494, "DEDR 4494 directly gives Parji poṭṭa ‘belly’ beside Kolami/Nayiki poṭṭa; the source's four poṭṭā responses keep its misspelled gloss ‘belley [sic]’ unchanged.")
add(17, 5247, "DEDR 5247 directly gives Parji vat (plural okul) ‘finger ring’ and Ollari vat; the Dhurwa vaṭ/vaṭṭu village variants denote the ring.")
add(23, 698, "DEDR 698 explicitly gives Parji ole(k) ‘house’ within the whole interior/house article; all three olek village responses match.")
add(37, 5422, "DEDR 5422 gives Parji vil ‘bow’ and Tamil vil, distinguishing this homonym from Parji vil ‘white’ in DEDR 5496.")
add(38, 178, "DEDR 178 directly gives Parji amb ‘arrow’ with cognate Tamil/Malayalam ampu; amb/ambu are the four village arrow responses.")
add(46, 5258, "DEDR 5258 gives Parji vāya ‘paddy field’ in a wider field article; the bāyā/vāyā village spellings retain this field noun.", forms={"bāyā", "vāyā"})
add(47, 1731, "DEDR 1731 directly gives Parji koppa ‘small hillock’ and related heap/mound forms; the four koppā ‘hill’ records match.", forms={"koppā"})
add(48, 4108, "DEDR 4108 directly gives Parji pāv ‘road’ beside Kolami pāv ‘path’; the four pāv ‘path’ records are exact.")
add(68, 2672, "DEDR 2672 gives Parji cupāri (plural cupārul) ‘tamarind’ and Ollari supār; cupāri/cupār are village variants.")
add(75, 1422, "DEDR 1422 directly gives Parji kākanḍi ‘brinjal’ beside Ollari kākal; the isolated kākāṇḍī/kākāndī variants fit.", forms={"kākāṇḍī", "kākāndī"})
add(92, 2200, "DEDR 2200 directly gives Parji kōḍ (plural kōḍul) ‘horn’, alongside Kannada/Tulu kōḍu; all four village forms match.")
add(98, 2682, "DEDR 2682 directly gives Parji codda ‘ant’ and closely related Gadaba/other Central Dravidian ant forms; the four coḍḍā responses match.")
add(100, 4791, "DEDR 4791 gives Parji mañja ‘man’ and Gondi manja ‘human being’; the four māñjā/māñyā ‘person’ responses fit.")
add(107, 2783, "DEDR 2783 directly gives Parji cālal (plural cālacil) ‘sister’, with Telugu cellelu and Gadaba cellāl; all four match.")
add(121, 410, "DEDR 410's whole proximal article includes Parji ine(n) in its ‘today’ subsection; all four ine ‘to day’ responses fit.")
add(122, 3516, "DEDR 3516 directly gives Parji tolli ‘tomorrow’ beside Kolami and Nayiki tolli; all four ṭollī records match.")
add(141, 2116, "DEDR 2116 directly gives Parji komaḍ ‘distant’ beside Nayiki kombaṛ; three kōmāḍ ‘far’ records match.", forms={"kōmāḍ"})
add(143, 4259, "DEDR 4259 directly gives Parji piṭit ‘little’ and Kannada puṭṭa ‘small’; the piṭīṭ variants are the four ‘small’ responses.")
add(154, 3655, "DEDR 3655 directly gives Parji nālu(k) ‘four things’ in the four-numeral family; all four nāluk/nālūk records match.")
add(174, 4641, "DEDR 4641 gives Parji mañ/mañj and mēndir ‘mist, dew’; the man/maṇ and meṇḍīr village variants retain these dew forms.")
add(177, 4452, "DEDR 4452 directly gives Parji botta ‘hole’ within the perforation/hole article; the three boṭṭā responses match.", forms={"boṭṭā"})
add(178, 4945, "DEDR 4945 directly gives Parji munḍa ‘tank’ beside Gadaba, Gondi, and Kui munḍa; the four mūnḍā/munḍā ‘pond’ responses fit.")
add(179, 5482, "DEDR 5482 directly gives Parji venḍra (usually plural venḍrel) ‘hair’; this isolated veṇḍrel response matches.", forms={"veṇḍrel"})
add(185, 1156, "DEDR 1156 directly gives Parji gaḍḍom ‘beard, moustache’ and Kolami gaḍḍam; the three gaddom ‘beard’ records fit.", forms={"gaddom"})
add(186, 1156, "DEDR 1156 directly gives Parji gaḍḍom ‘beard, moustache’; the two isolated gaddom ‘mustache’ records fit.", forms={"gaddom"})
add(188, 2002, "DEDR 2002 directly gives Parji kēndid (plural kēndil) ‘feather’ beside Gadaba kendiṭ; three keṇḍīḍ records match.", forms={"keṇḍīḍ"})
add(190, 4083, "DEDR 4083(c) directly gives Parji pilpili and Nayiki pipuli ‘butterfly’; isolated pīlpīlī matches.", forms={"pīlpīlī"})
add(191, 692, "DEDR 692 directly gives Parji ḍū (plural ḍuvul) ‘tiger’; the two ḍūv/ḍūvu responses preserve this stem.", forms={"ḍūv", "ḍūvu"})
add(192, 857, "DEDR 857 directly gives Parji ili (plural ilil) ‘bear’, beside Telugu elūgu and Gadaba ilij; all four ilī responses match.")

source_conflicts = {6, 7, 8, 9, 13, 20, 25, 28, 39, 42, 55, 56, 59, 71, 73,
                    91, 99, 138, 139, 148, 156, 157, 158, 159, 160, 163,
                    171, 175, 179, 182, 188, 190, 191, 200}
loanlike = {7, 15, 16, 18, 19, 20, 24, 26, 32, 33, 36, 56, 57, 58, 68,
            69, 70, 74, 76, 77, 78, 80, 87, 89, 94, 112, 113, 114, 124,
            126, 129, 149, 150, 160, 180, 185, 186, 194, 196, 197, 199, 200}
compounds = {9, 18, 20, 26, 28, 39, 46, 68, 75, 99, 118, 124, 125, 127,
             128, 138, 139, 148, 156, 157, 158, 159, 160, 163, 167,
             169, 171, 175, 176, 179, 182, 186, 190, 199, 200}

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
    if n == 24:
        reason = f"{form!r} ‘{gloss}’ resembles regional kapāṭ door words, but the global comparison does not establish the immediate donor into Dhurwa Parji."
    elif n == 26:
        reason = f"{form!r} ‘{gloss}’ resembles the regional kiḍki window family (CDIAL 3770); its immediate donor and the spaced variant's segmentation are unresolved."
    elif n == 36:
        reason = f"{form!r} ‘{gloss}’ matches Parji nã̄gil in DEDR 2907, whose full article also compares Indo-Aryan lāṅgala-/naṅgala-; contact direction and immediate donor require resolution."
    elif n == 42:
        reason = f"{form!r} ‘{gloss}’ resembles Telugu nīḍa in DEDR 3679, while that article gives older Parji nīṛa; the Dhurwa stop versus rhotic and possible donor path are unresolved."
    elif n == 57:
        reason = f"{form!r} ‘{gloss}’ resembles the regional pītal brass family (CDIAL 8184), but its immediate donor into this Parji list is not verified."
    elif n == 89:
        reason = f"{form!r} ‘{gloss}’ matches Parji cir in DEDR 2554, but that article compares Sanskrit/Pkt. buffalo words; the immediate contact path and relation type need resolution."
    elif n == 196:
        reason = f"{form!r} ‘{gloss}’ matches Parji caṭṭa in DEDR 2311, whose article also points to CDIAL 4574 *caṭṭa- ‘bamboo matting’; the immediate donor is unresolved."
    elif n == 171 and form == "ṭōlu":
        reason = f"{form!r} ‘{gloss}’ resembles DEDR 3559 Parji tōl ‘skin, bark’ and Telugu tōlu, but the Dhurwa initial retroflex stop is unexplained; direct inheritance versus contact remains open."
    elif n in source_conflicts:
        reason = f"Appendix B item {n} records {form!r} ‘{gloss}’, but the elicited sense or segment boundary needs source-level verification."
    elif n in loanlike:
        reason = f"{form!r} ‘{gloss}’ is loanlike or shared regional vocabulary; an immediate donor into this Parji variety has not been established."
    elif n in compounds or " " in form:
        reason = f"{form!r} ‘{gloss}’ is compound or multiword; a full secure component analysis is lacking."
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
