"""Draft Alu Kurumba Varghese B3 analyses for review, without saving edges."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 100

# One-based residual inventory index -> (DEDR article, primary comparison).
# Each DEDR node is a whole article; the displayed head need not spell the
# source reflex exactly. Elucidating donor paths requires separate evidence.
proposals = {
    1: (4682, "DEDR 4682 has Kota maṇḍ 'head', Kannada maṇḍe and Koraga maṇḍa; the survey maṇḍa matches both form and sense."),
    2: (5352, "DEDR 5352 records Kannada bāy(i) and Kodagu ba·y 'mouth', alongside Tamil and Malayalam vāy; survey bāyɨ preserves the Kannada-type initial."),
    3: (3986, "DEDR 3986 has Kannada pal and Kodagu pallï 'tooth'; nearby survey lects attest pallɨ. Alu Kurumba allɨ shows initial p-loss, also seen in its āḷɨ 'milk'."),
    4: (3633, "DEDR 3633 explicitly gives Kannada nālage/nālige and Kota na·lg 'tongue'; nāḷaŋge is a close local continuation."),
    11: (4559, "DEDR 4559 gives Kannada pottu/hottu 'time, sun' and Badaga ottu 'sun'; the survey ottu matches the latter directly."),
    12: (4016, "DEDR 4016 has Irula paḷḷa 'river', Kannada paḷḷa 'stream' and Badaga haḷḷa 'stream, river'; allā shares the local river sense with initial loss."),
    15: (1514, "DEDR 1514 lists Kannada kiccu and Tamil kiccu 'fire', with Kurumba ticci; survey kicɨ belongs to this fire family."),
    17: (4345, "DEDR 4345 lists Tamil pū and Malayalam pū/pūvu 'flower'; Alu Kurumba ū has the same vowel and meaning with initial p-loss."),
    26: (4096, "DEDR 4096 lists Kannada pāl and Kodagu pa·lï 'milk'; Alu Kurumba āḷɨ agrees with neighboring -ɨ forms apart from its recurrent initial p-loss."),
    27: (3650, "DEDR 3650 records Tamil/Malayalam nāy and Kannada nāyi 'dog'; the survey nāyɨ directly matches."),
    28: (4085, "DEDR 4085 lists Kannada pāvu and Badaga āvu/hāvu 'snake'; the survey āvu exactly matches the local Badaga variant."),
    30: (4410, "DEDR 4410 gives Kannada pesar(u), Belari hudari and Koraga hudari 'name'; survey yejaru is a local variant of that name noun."),
    33: (183, "DEDR 183 includes Gadhaba ammi 'younger sister'; Jambu's Badaga amme has exactly that kin sense, supporting the Alu Kurumba amme."),
    34: (1873, "DEDR 1873 lists Kannada kūsu 'infant, babe' and Koraga kūje 'small boy'; the survey kūsu 'son' is a kinship extension."),
    35: (4395, "DEDR 4395 groups Tamil peṇ 'daughter, girl' with Kannada heṇṇu 'female'; survey ennu 'daughter' fits the h-/p-less local variant."),
    36: (4395, "DEDR 4395 directly gives Kannada peṇḍati 'wife'; yinḍrɨ retains the nasal-stop female stem used for wives in this article."),
    44: (4547, "DEDR 4547 lists Malayalam pollā/pollāta 'bad' and Kannada polla 'badness'; Badaga olla has the same sense and supports survey oḷḷā."),
    50: (5496, "DEDR 5496 has the veḷ-/beḷḷ- white family, including Badaga beḷḷe 'white'; this is the first of two separately identified survey belle records."),
    51: (5496, "DEDR 5496 has Badaga beḷḷe and Kannada beḷḷ- 'white'; this second source-defined belle record remains distinct."),
    52: (3748, "DEDR 3748 lists Kannada nettara 'blood' and Gondi nettur 'blood, red'; survey nettā ru 'red' uses the same blood-to-red sense attested in the article."),
    53: (474, "DEDR 474 lists Kannada eraḍu and related Dravidian forms for 'two'; survey yeraḍu is the expected local form."),
    54: (3655, "DEDR 3655 directly records Kannada nāku as a variant of nālu 'four'; survey nākku matches it."),
    56: (5151, "DEDR 5151 includes Tamil ār and Kannada ār 'who'; survey ārā is a local interrogative form."),
    57: (5151, "DEDR 5151 gives Kannada enna 'what kind' and Malayalam entu 'what'; survey yenadɨ is an interrogative neuter form."),
    58: (5151, "DEDR 5151 groups en-/e- interrogatives; the identical second yenadɨ 'what' record retains its separate source identity."),
    59: (5151, "DEDR 5151 directly lists Kannada elli 'where'; survey elli is an exact form-and-sense match."),
    61: (5151, "DEDR 5151 gives Kannada eṣṭu and ēsu 'how many'; survey ɛśɨhɨ is a local interrogative-quantity form."),
    63: (410, "DEDR 410(a) records Kannada idu and Malayalam itu 'this'; survey idɨ is the proximal neuter demonstrative."),
    64: (1, "DEDR 1 records Kannada adu and Malayalam atu 'that'; survey adɨ is the distal neuter demonstrative."),
    71: (3263, "DEDR 3263 gives Tamil tiṉ and Malayalam/Kannada tinn- 'eat'; tinnuṭottā contains this eat stem with elicited verbal morphology."),
    72: (1097, "DEDR 1097 lists Kannada kaccu and the earlier Alu Kurumba kacu 'bite'; kaciṭottā has the same stem with verbal morphology."),
    74: (1654, "DEDR 1654 lists Kannada kuḍi and earlier Alu Kurumba kuḍi 'drink'; kuḍiddɨdiyā is an inflected continuation of that verb."),
    79: (1835, "DEDR 1835 records Kannada kuḷḷu 'sit down' and prior Alu Kurumba kuḷi; kuḷiduṭṭattā preserves the sit stem."),
    84: (1041, "DEDR 1041 groups Kannada ōḍu 'run' with prior Alu Kurumba oḍu; oḍiṭattu retains the same run stem."),
    85: (4572, "DEDR 4572 includes the pō-/pōy- go paradigm; prior Alu Kurumba ogu and this oyɨṭatu both belong to the go family."),
    86: (5270, "DEDR 5270 lists Kannada bar- (band-) and Tamil varu (vant-) 'come'; banduṭattu has the attested nasal past stem."),
    87: (3784, "DEDR 3784 records Kannada nuḍi 'speak' and Telugu noḍi 'word'; noḍiduṭāttu preserves the speech stem."),
    89: (2017, "DEDR 2017 gives Tamil kēḷ (kēṭṭ-) and Kannada kēḷ 'hear'; ketu ṭattu retains the past hear stem in an elicited verbal expression."),
    90: (3794, "DEDR 3794 directly records Kannada nōḍu 'look, see'; survey noḍu is the same verb."),
    91: (3794, "DEDR 3794 records Kannada nōḍu 'look'; survey noḍiyā is a separately recorded inflected variant."),
    92: (5160, "DEDR 5160 gives Tamil nāṉ and Kannada nān 'I'; the survey nān is the same first-person singular pronoun."),
    93: (3684, "DEDR 3684 lists Kannada nī(n) and Tamil/Malayalam nī 'you (sg.)'; survey ni is the informal singular record."),
    94: (3684, "DEDR 3684 has the singular second-person nī family; the identical formal ni record differs in usage, not lexical ancestry."),
    95: (1, "DEDR 1 lists Kannada ava and Tamil avaṉ 'he'; survey avã is the distal masculine pronoun."),
    96: (1, "DEDR 1 lists Kannada avaḷ and Malayalam avaḷ 'she'; survey ava is the local feminine distal pronoun."),
    98: (5154, "DEDR 5154 specifically lists Kodagu naŋga and Tamil nāṅkaḷ in the exclusive 'we' family; the source labels this record exclusive."),
    99: (3688, "DEDR 3688 gives Kannada nīngaḷ and Kodagu niŋga 'you (pl.)'; survey niŋga matches this plural."),
    100: (1, "DEDR 1 lists Kannada avar 'those persons, they'; survey avarɨ is the third-person plural distal pronoun."),
}

# Eleven, these and those transparently contain independently attested pieces.
components = {
    55: (["d3918", "d990"], "The survey annondu 'eleven' combines the DEDR 3918 ten/ann- numeral base with DEDR 990 Kannada ondu 'one'; preserve the order ten + one."),
    65: (["d410", "d844"], "idellā̃ 'these' combines proximal id- (DEDR 410, Kannada idu) and ellā 'all' (DEDR 844, Kannada ella); both parts are represented in source order."),
    66: (["d1", "d844"], "adellā̃ 'those' combines distal ad- (DEDR 1, Kannada adu) and ellā 'all' (DEDR 844); this is the separately recorded distal plural."),
}

holds = {
    5: "vɸḷḷaŋgayɨ 'palm' resembles an 'inner hand' expression, but its first element and exact compound segmentation remain unclear.",
    6: "aḍilɨ 'door' matches nearby Irula paḍīlɨ, but no secure DEDR article or donor path was found.",
    7: "cavɨde 'firewood' has no verified DEDR stem or immediate donor.",
    8: "bārɨ 'broom' has no securely resolved DEDR family; similarity to broom terms elsewhere is not enough.",
    9: "koḍāli 'axe' is shared regional tool vocabulary with a possible Indo-Aryan comparison; immediate donor remains uncertain.",
    10: "modɨrā 'ring' resembles Malayalam mōtiram/contact vocabulary, but the immediate donor has not been verified.",
    13: "kānābillɨ 'rainbow' appears compounded ('sky bow'), but both exact components and their parent nodes need resolution.",
    14: "Second independent kānābillɨ 'rainbow' source record; the proposed compound remains unanalysed.",
    16: "DEDR 4316 brackets Kannada būdi 'ash' and explicitly suggests borrowing from Sanskrit bhūti; the immediate donor is unresolved.",
    18: "gōdume 'wheat' appears to be crop/contact vocabulary; its immediate donor and inherited status are unresolved.",
    19: "urɨḷagāsɨ 'potato' may contain a tuber term, but compound pieces and recent crop contact need verification.",
    20: "acamāsɨ 'chilli' lacks a securely segmented stem or identified immediate donor.",
    21: "beŋgāyɨ 'onion' is shared regional crop vocabulary; an exact ancestry and contact path remain uncertain.",
    22: "ūgōsu 'cauliflower' is recent crop vocabulary; its pieces and immediate donor need verification.",
    23: "to mbaṇṇu 'tomato' is a spaced recent-crop response without a verified donor or segmentation.",
    24: "gōsu 'cabbage' is shared crop vocabulary; the immediate donor has not been established.",
    25: "danā 'cow' is shared with Badaga and Toda but lacks a secure DEDR article and immediate donor analysis.",
    29: "suhune 'mosquito' has no secure article-level lexical match.",
    31: "manṣan 'man' looks like a manuṣya-type loan; the immediate donor is not identified.",
    32: "ɨṇɨgā 'woman' resembles DEDR 4395 Toda iṇ and Kannada heṇṇiga, but the latter means 'impotent man, coward'; the suffix and semantic route need clarification.",
    37: "jinā 'day' is shared with Badaga and resembles Indo-Aryan dina; immediate donor remains unresolved.",
    38: "orakkudu 'morning' is apparently morphologically complex; a sleep-based comparison alone does not explain the whole form.",
    39: "santu 'evening/afternoon' resembles sandhyā contact vocabulary, but the immediate donor is unverified.",
    40: "næyagu 'tomorrow' could continue a nāḷ- day formation, but its segments and sound correspondence remain unclear.",
    41: "bārā 'week' is likely vāra-type calendar vocabulary; the immediate donor is not verified.",
    42: "varusā 'year' resembles a varṣa-type loan; the immediate donor is not verified.",
    43: "oyiadu 'old' has no securely identified article-level comparison.",
    45: "nannidollā 'wet' may be a negative/adjectival expression; the apparent oḷḷā part cannot alone account for the whole form.",
    46: "balabakkā 'right' contains a likely bala right component, but the pakkā 'side' component lacks a verified parent.",
    47: "aḍapakkɨ 'left' seems compounded with a side noun; exact source segmentation and the second parent are unresolved.",
    48: "dūrā 'far' resembles Indo-Aryan dūra; the immediate donor is unresolved.",
    49: "bārā 'heavy' differs in meaning from the source's identical week word and resembles bhāra-type loan vocabulary; do not collapse these homonyms.",
    60: "yɛgɨ 'when' may reflect an interrogative e-/y- base, but its suffix and exact formation are not identified.",
    62: "ɛnavaga 'what kind' appears to include interrogative ena-, but the second component is unresolved.",
    67: "ondemaggɨ 'same' may begin with ondu 'one'; the remaining material is not identified.",
    68: "mukkai 'whole' resembles other mur̤u-/muḷu- whole terms, but the final consonants and exact etymon are unclear.",
    69: "unnā 'few' lacks a secure same-sense lexical comparison.",
    70: "appārā 'many' matches Badaga/Irula appara but has no verified DEDR parent or borrowing path.",
    73: "oṭṭeasitɨoṭotā 'he is/was hungry' is a multi-part predicate; no single stem accounts for the whole response.",
    75: "arddɨdāve 'he was thirsty' includes an auxiliary or tense element; the thirsty stem and full segmentation remain unresolved.",
    76: "arddɨ 'thirsty' has no securely identified article-level lexical parent.",
    77: "biddo 'lie down, he lay down' could contain a verbal stem but has no verified matching DEDR article.",
    78: "biddiṭoddā is the second lie-down source response; its root and inflection cannot yet be separated securely.",
    80: "kicoṭṭu iḍidɨtā avā 'it burns/it burned' is a multiword clause; linking only kic- 'fire' would flatten the response.",
    81: "sāyaveḍā combines a negative 'don't die' and past 'he died' elicitation; the full expression and negative element remain unresolved.",
    82: "kolavaḍa combines 'don't kill' and 'he killed'; DEDR 2132 explains kol- but not the whole negative/past response.",
    83: "parantu ḍodu 'fly, it flew' is a multiword form with a second element not securely analysed.",
    88: "ɤ̄gɨ 'listen' does not securely match the kēḷ-/kēṭ- stem of the second response; withhold a speculative parent.",
    97: "naŋga is labelled inclusive here, but DEDR 5154 treats exact Kodagu naŋga as exclusive; the identical exclusive survey record is assessed separately.",
}
assert not (set(proposals) & set(components))
assert not ((set(proposals) | set(components)) & set(holds))
assert set(proposals) | set(components) | set(holds) == set(range(1, len(records) + 1))

accepted = []
for ix, (num, evidence) in proposals.items():
    rec = records[ix - 1]
    accepted.append({"record": rec, "parent": f"d{num}", "kind": "reflex",
                     "citation": f"{rec['Source']};dedr[entry {num}]", "evidence": evidence,
                     "status": "pending-review", "surveyIndex": ix})
for ix, (parts, evidence) in components.items():
    rec = records[ix - 1]
    accepted.append({"record": rec, "parent": parts[0], "components": parts, "kind": "component",
                     "citation": rec["Source"] + ";" + ";".join(f"dedr[entry {p[1:]}]" for p in parts),
                     "evidence": evidence, "status": "pending-review", "surveyIndex": ix})
accepted.sort(key=lambda x: x["surveyIndex"])
held = [{"record": records[ix - 1], "reason": reason, "status": "research-hold", "surveyIndex": ix}
        for ix, reason in sorted(holds.items())]
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")

parents = {p for x in accepted for p in x.get("components", [x.get("parent")])}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, sorted(parents - set(articles))
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
owners = {x["record"]["ID"]: "proposed" for x in accepted}
owners.update({x["record"]["ID"]: "held" for x in held})
assert len(owners) == len(records) and set(owners) == {r["ID"] for r in records}
(P / "record-coverage-001.json").write_text(json.dumps({"source": "varghese2015palakkad", "language": "AluKurumba",
    "residualSourceRecords": len(records), "proposedRecords": len(accepted), "heldRecords": len(held),
    "unexaminedRecords": 0, "uniqueOwnership": True, "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"proposedRecords": len(accepted), "assignmentRows": sum(len(x.get("components", [x.get("parent")])) for x in accepted),
                  "heldRecords": len(held), "parents": len(parents)}))
