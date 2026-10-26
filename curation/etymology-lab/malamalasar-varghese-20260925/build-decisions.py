"""Draft Mala Malasar comparative assignments for user review only."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())

# Index is the one-based position in survey-inventory.json. DEDR is cited by
# complete article, including the exact language-specific comparandum.
proposals = {
    1: (5031, "DEDR 5031 lists Malayalam mōnta 'face' and Kota mo·nd 'face'; the survey mɔndai preserves the same moṇḍ-/mond- face family."),
    2: (1448, "DEDR 1448 directly lists Malayalam kātu 'ear' and Irula ka·du; survey kātɨ is the corresponding ear noun."),
    3: (5352, "DEDR 5352 gives Malayalam vāy and Kannada bāy(i) 'mouth'; survey bāyɨ matches the latter consonant shape."),
    4: (5409, "DEDR 5409 gives Malayalam viral and Kannada beral, beside Tulu birelů, 'finger'; survey bira l belongs to this finger family."),
    6: (2007, "DEDR 2007 explicitly gives Tamil cēri 'village' and Malayalam cēri 'village street'; the survey form and meaning fit this article. The article also notes an Indo-Aryan comparison, so contact remains a caveat."),
    12: (3774, "DEDR 3774 gives Malayalam nēram 'sun, day, light, time' and Kodagu ne·ra 'sun, time'; survey nerā is a close sun reflex."),
    13: (3045, "DEDR 3045 directly lists Tamil taṇṇi and Malayalam taṇṇi 'water'; the survey's spaced transcription ta ṇṇi represents the same lexical item."),
    19: (4867, "DEDR 4867 lists Malayalam muḷaku 'pepper' and Kota meḷg and Tulu muṇci with 'chilli' senses; survey muḷakɨ 'chilli' is the southern pepper-to-chilli extension."),
    24: (399, "DEDR 399 groups Tamil āṇ 'male' and āḷ 'man' with Malayalam āṇ/āḷ and Kodagu a·ṇ a·ḷï 'man'; survey āṇaḷɨ matches the compound shape in this whole article."),
    35: (4888, "DEDR 4888 gives Malayalam mukaḷ 'top, summit' and cognate 'top' terms elsewhere; survey mukaḷɨ 'above' is an adverbial use of that top noun."),
    36: (5496, "DEDR 5496(a) directly lists Malayalam veḷḷa 'white colour'; survey vaḷḷa has the same consonantal frame and meaning."),
    37: (5052, "DEDR 5052 gives Malayalam mūnnu, Kota mu·nḏ and Kodagu mu·ndï 'three'; survey mūntɨ is the same numeral family."),
    39: (5151, "DEDR 5151 explicitly includes Tamil ār and Malayalam ār 'who'; survey ārɨ is an inflected local form of the interrogative."),
    40: (5151, "DEDR 5151 lists Tamil eṉṉa/eṉṉatu 'what' and Malayalam entu 'what'; survey ennatɨ preserves the interrogative base and neuter ending."),
    41: (5151, "DEDR 5151 lists Tamil eṉṉa/eṉṉatu 'what' and Malayalam entu 'what'; this second source-defined record has the same survey form."),
    42: (5151, "DEDR 5151 gives Tamil eṅkē and Malayalam eṅṅu 'where'; survey ɛŋgɛ shares that interrogative-place formation."),
    44: (5151, "DEDR 5151 lists Tamil eppōtu and Malayalam eppōḷ 'when'; survey ɛppam is a local temporal interrogative in that article family."),
    45: (5151, "DEDR 5151 directly gives Malayalam etra 'how many, how much'; the survey's spaced ɛtta ra reflects this interrogative quantity word."),
    46: (410, "DEDR 410(a) lists Malayalam itu and Kodagu id 'this'; survey idɨ is the proximal neuter demonstrative."),
    47: (1, "DEDR 1 lists Malayalam atu and Kodagu ad 'that'; survey adɨ is the distal neuter demonstrative."),
    52: (1124, "DEDR 1124 gives Malayalam kaṭikka and Kannada/Kodagu kaḍi 'bite'; survey kaḍi belongs to this bite verb, with ordinary elicited verbal morphology."),
    57: (2132, "DEDR 2132 directly gives Malayalam kolluka 'kill'; survey kollatɨ has the same kill stem, though the source gloss combines negative imperative and past contexts."),
    58: (4091, "DEDR 4091(a) gives Tamil pār 'see, look' and Malayalam pārkka 'regard, behold'; survey parɨ is the elicited look/see stem."),
    59: (5160, "DEDR 5160 gives Tamil nāṉ 'I' and the first-person singular forms across Dravidian; survey nān is the same pronoun."),
    60: (3684, "DEDR 3684 gives Tamil nī and Malayalam nī 'you (sg.)'; survey niyɨ is the singular pronoun in the informal source slot."),
    61: (3684, "DEDR 3684 gives Tamil nī and Malayalam nī 'you (sg.)'; the identical niyɨ in the formal source slot appears to differ pragmatically rather than etymologically."),
    62: (1, "DEDR 1 explicitly lists Malayalam avan 'that man, he'; survey avan matches the masculine distal pronoun."),
    63: (1, "DEDR 1 explicitly lists Malayalam avaḷ 'that woman, she'; survey avaḷɨ is the feminine distal pronoun."),
    64: (3647, "DEDR 3647 gives Malayalam nām 'we, I and you' with inclusive interpretation; the survey explicitly labels nām inclusive."),
    66: (3688, "DEDR 3688 lists Tamil nīvir 'you (pl.)' alongside other Dravidian second-person plurals; survey niviru is the same plural formation."),
    67: (1, "DEDR 1 explicitly lists Malayalam avar 'those persons'; survey avar is the third-person plural distal pronoun."),
}

reasons = {
    5: "nɛgam resembles the widespread nakha/nakham fingernail word; immediate donor and exact ancestry remain unresolved.",
    7: "cāḷa 'house' may belong to śālā/cālai contact vocabulary; no secure immediate donor or DEDR family identified.",
    8: "ara kallɨ 'mortar' is a multiword response; its ara and stone components need exact analysis before ordered edges.",
    9: "kuḍuŋgallɨ 'pestle' appears compounded with kallɨ 'stone'; the first element and lexical sense are not securely resolved.",
    10: "DEDR 1722 compares Dravidian guddali-type tools with Sanskrit kuddāla; koḍāli may be a regional tool loan, with immediate donor unresolved.",
    11: "modira m 'ring' resembles Malayalam mōtiram, a contact term; no verified immediate donor node.",
    14: "karaṇam baḷaykatɨ 'rainbow' is a multiword descriptive response whose components are not securely identified.",
    15: "pɛrivai 'path' might compress peru 'big' plus var̤i/vāzhi 'way', but segmentation and correspondence are uncertain.",
    16: "cāram 'ash' is shared locally but does not securely match DEDR 2453 cāmpal without an unexplained nasal/stop loss.",
    17: "pacaram 'mud' has no secure lexical parent; similarity to cāram 'ash' is insufficient.",
    18: "kota mbɨ 'wheat' resembles a widespread borrowed wheat term; immediate donor and segmentation unresolved.",
    20: "bɛŋgayam 'onion' is regionally widespread but an exact DEDR ancestry and immediate donor remain unresolved.",
    21: "ta kkāḷi 'tomato' names a post-Columbian crop and may be a compound/contact term; no direct inherited whole-word proposal.",
    22: "mɔṭakosɨ 'cabbage' is shared among nearby survey lects, but its contact path and immediate donor are unresolved.",
    23: "pāravala 'spider' lacks a secure match to DEDR spider terms; apparent vala 'web' is insufficient for whole-word analysis.",
    25: "DEDR 196(a) has Tamil tam-aiyaṉ 'elder brother' and Malayalam ayyan 'father, lord'; bare ayyan 'older brother' is a possible kin extension but needs source-specific confirmation.",
    26: "pannaṇḍumaṇi 'noon' is 'twelve o'clock' in form; numeral and time-unit components need separately verified targets.",
    27: "vāra m 'week' is an Indo-Aryan-style calendar word with uncertain immediate donor.",
    28: "māsam 'month' is a likely Indo-Aryan loan; exact immediate donor is unresolved.",
    29: "Second source-defined māsam 'month' record; likely loan with unresolved immediate donor.",
    30: "barśam 'year' is a likely varṣa-type loan; immediate donor is unresolved.",
    31: "Second source-defined barśam 'year' record; immediate donor remains unresolved.",
    32: "mośa 'bad' resembles contact vocabulary; no secure immediate donor or inherited cognate.",
    33: "dūrā 'far' resembles Indo-Aryan dūra and may be borrowed; immediate donor unresolved.",
    34: "paḷanunɨ 'light' has no securely identified lexical stem in DEDR.",
    38: "padinontɨ 'eleven' is morphologically ten plus one, but exact inflection and component targets are not verified.",
    43: "ɛkkaḍɛ 'where' resembles Telugu ekkaḍa; immediate donor or inherited construction unresolved.",
    48: "ontɨpila 'same' contains a probable 'one' base, but the remaining element is not identified.",
    49: "vitv eṣam 'different' resembles a loan of Sanskrit vyatyāsa; immediate donor is unresolved.",
    50: "oḷḷik 'few' has no securely identified parent family.",
    51: "oniyama 'many' has no securely identified parent family.",
    53: "paṭṭiṇi piḍikkatɨ is a multiword predicate 'be hungry'; the source gloss also combines aspect values.",
    54: "ta ṇṇi viḍakkatɨ is a multiword thirsty predicate; water alone cannot stand as its whole etymology.",
    55: "ti vekutɨ is a multiword 'burn' predicate; both words need analysis.",
    56: "cakata carries contradictory 'don't die, he died' elicitation contexts; negative and stem require disambiguation.",
    65: "namɨ is labelled exclusive though DEDR 3647 nām/nam- is inclusive; neither the morphological nor source-label conflict is resolved.",
}
assert set(proposals) | set(reasons) == set(range(1, len(records)+1))
assert not (set(proposals) & set(reasons))

accepted = []
for ix, (num, evidence) in proposals.items():
    rec = records[ix-1]
    accepted.append({"record": rec, "parent": f"d{num}", "kind": "reflex",
        "citation": f"{rec['Source']};dedr[entry {num}]", "evidence": evidence,
        "status": "pending-review", "surveyIndex": ix})
held = [{"record": records[ix-1], "reason": reason, "status": "research-hold", "surveyIndex": ix}
        for ix, reason in reasons.items()]
accepted.sort(key=lambda x: x["surveyIndex"])
held.sort(key=lambda x: x["surveyIndex"])
(P/"proposed-decisions.json").write_text(json.dumps({"status":"pending-review","accepted":accepted,"held":held},ensure_ascii=False,indent=2)+"\n")
(P/"research-holds.json").write_text(json.dumps(held,ensure_ascii=False,indent=2)+"\n")
(P/"unexamined-inventory.json").write_text("[]\n")
parents = {x["parent"] for x in accepted}
articles = {}
with gzip.open(P.parents[2]/"data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k:row[k] for k in ["Page","Citation","Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page="+row["Page"]
assert set(articles) == parents
(P/"primary-evidence.json").write_text(json.dumps(articles,ensure_ascii=False,indent=2)+"\n")
source_ids = [r["ID"] for r in records]
owners = {x["record"]["ID"]: "proposed" for x in accepted}
owners.update({x["record"]["ID"]: "held" for x in held})
assert set(owners) == set(source_ids) and len(owners)==len(source_ids)
(P/"record-coverage-001.json").write_text(json.dumps({"source":"varghese2015palakkad",
    "language":"MalaMalasar", "residualSourceRecords":len(records),
    "proposedRecords":len(accepted),"heldRecords":len(held),"unexaminedRecords":0,
    "uniqueOwnership":True,"recordOwnership":owners},ensure_ascii=False,indent=2)+"\n")
print(f"{len(accepted)} proposed records, {len(held)} holds")
