"""Conservative, research-only Badaga Census Tamil Nadu etymology proposals."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 421
proposals = {}


def add(indices, parent, evidence):
    for index in indices:
        assert index not in proposals
        proposals[index] = (parent, evidence)


# Every cited DEDR number below was checked in the complete cached article.
add([14], 3170, "DEDR 3170 lists Kannada dāri 'way, road, path' alongside Tamil tāri and Tulu dāri; the Census road form is exact.")
add([19], 3045, "DEDR 3045 has Tamil taṇṇīr and Malayalam taṇṇīr 'water'; Hockings' Badaga taṇṇīru 'water, fresh water' is already linked to the same article.")
add([27], [4395, 1873], "The two lexical elements are female + child: DEDR 4395 has Badaga heṇṇu 'woman/female' and 1873 has Badaga kusu 'child'; the Census concatenation hɛṇṇukūśu is transparent.")
add([29], 1173, "DEDR 1173 lists Kota gaṇḍ 'male' and Kannada gaṇḍu 'the male sex, a male, man'; Hockings' Badaga gandu 'boy, man' confirms the local shape.")
add([38, 144], 621, "DEDR 621 compares Kannada udda 'height, length, depth' and Kodagu udda 'length, height'; both Census uddə̄ 'tall' records carry the same stem.")
add([39], 4228, "DEDR 4228 gives Tulu pīli 'dry, thin, light' and Telugu pīla 'thin, lean'; Hockings has Badaga pīlu 'thin, small, tiny'.")
add([49], 4395, "DEDR 4395 lists Kannada heṇṇu 'woman, female' and the Badaga female stem; the Census hɛṇṇuṛu 'wife' has a local ending on that stem.")
add([52], 2170, "DEDR 2170 has Kannada kotti 'male or female cat'; Hockings independently records Badaga kotti 'domestic cat'.")
add([54], 2248, "DEDR 2248 gives Kota ko·y 'fowl' and explicitly cites Badaga kó·í; Census kɔ̄yi 'cock' is the same local bird noun.")
add([60], 3606, "DEDR 3606 lists Kannada nari 'jackal, fox' and Tulu nari 'jackal, fox'; Hockings' Badaga nari is specifically 'Indian fox, Bengal fox'.")
add([62], 4642, "DEDR 4642 lists Tamil/Malayalam mayil 'peacock' and Kannada maylu; Census məiyilu matches this avian family, though the article also notes Sanskrit mayūra-.")
add([64], 1711, "DEDR 1711(a) lists Tamil kutirai and Kannada kudure 'horse'; Census kutərə retains that horse stem with local voiceless medial transcription.")
add([68], 4307, "DEDR 4307 lists Tamil/Malayalam puli and Kannada puli 'tiger', and explicitly says Toda üly is from Badaga huli; Hockings' Badaga huli confirms the local loss of initial h in Census uḷi.")
add([75], 4818, "DEDR 4818 lists Tamil mār 'breast, chest' and Malayalam māṟu 'chest'; Hockings' Badaga māru 'breast, chest' is an exact local comparison.")
add([81], 4682, "DEDR 4682 has Kota maṇḍ, Kannada maṇḍe and Kodagu maṇḍe 'head'; the Census maṇḍə̃ has the same head stem.")
add([87], 3633, "DEDR 3633 lists Kannada nālage/nālige 'tongue' and Kota na·lg; Census nāleŋgə̃ preserves the extended tongue stem.")
add([88], 3986, "DEDR 3986 lists Tamil pal and Kannada pal 'tooth'; Hockings' Badaga hallu 'tooth, teeth' matches the Census form exactly.")
add([92], 4096, "DEDR 4096 lists Kannada pāl 'milk'; Hockings has Badaga hālu 'milk', the expected Badaga h-initial counterpart to Kannada p-.")
add([93], 215, "DEDR 215 has Kota aky 'husked grain' and Kannada akki 'husked rice'; Hockings' Badaga akki is also 'husked rice'.")
add([111], 4345, "DEDR 4345 has Tamil pū, Kota pu and Kannada pūvu 'flower'; Census ū is the shortened Badaga flower form, with Hockings' hū as a local variant.")
add([113], 4300, "DEDR 4300 lists Kannada pul 'grass, straw' and explicitly Toda ul 'grass' from Badaga; Hockings' Badaga ullu/hullu 'grass, hay' matches Census ullu.")
add([118], 3753, "DEDR 3753 gives Tamil nel, Kota nel and Kannada nel/nellu 'paddy, unhusked rice'; Census nɛllu matches the Kannada form.")
add([119], 2815, "DEDR 2815 lists Tamil ēr 'plough' and Kannada ēru 'pair of oxen yoked to a plough'; Hockings' Badaga ēru is a plough-set.")
add([121], 4711, "DEDR 4711(a) lists Tamil maram and Kannada mara 'tree'; Census moṛə̄ is a Badaga tree form with the matching mar-/mor- stem.")
add([141], 5496, "DEDR 5496(a) lists Tamil veḷ/veṇ and Kannada biḷi 'white'; the Census bɛḷḷə retains the white stem, distinct from homophonous 'silver'.")
add([167], 1278, "DEDR 1278(a) explicitly lists Kannada kabbiṇa 'iron' in the black/charcoal family; Census kabbuñə̄ is a close Badaga iron form, like Badaga LSI kabbuṇa.")
add([170], 5496, "DEDR 5496(a) lists Tamil veḷḷi 'silver' and Kannada beḷḷi 'silver'; Hockings' Badaga belli is the same metal noun.")
add([174], 474, "DEDR 474 lists Tamil iraṇṭu, Malayalam raṇṭu and Kannada eraḍu 'two'; Census ɛṛəṭu fits the dental/retroflex numeral alternation.")
add([175], 3655, "DEDR 3655 lists Tamil nāṉku and Kannada nālku 'four'; Census nākku reflects a local reduction of the lateral cluster.")
add([176], 2826, "DEDR 2826 directly lists Kannada ay(i)du 'five'; Census aitu is the same numeral with dental voicing/transcription variation.")
add([177], 910, "DEDR 910 lists Kannada ēr̤u 'seven'; Hockings' Badaga iḷu 'seven' confirms the local reduced form.")
add([185], 4797, "DEDR 4797 directly lists Kannada māḍu 'to do, make' and notes Kota ma·ṛ- from Badaga; Census māṭɔ̄tu is the do verb with infinitive ending.")
add([192], 3570, "DEDR 3570 lists Tamil nakku, Kannada nakku and Kota nak- 'to lick'; Census nakkɔ̄tu adds the elicited infinitive ending.")
add([202], 4834, "DEDR 4834 lists Tamil māṟu 'sell, exchange' and Kannada māru 'sell'; Census mārɔ̄tu is the same verb with infinitive ending.")
add([203], 4915, "DEDR 4915 lists Kannada muccu 'close, shut, cover' and Kota muc- 'cover'; Census muccɔ̄tu retains the shut stem.")
add([238], 2053, "DEDR 2053 has Kannada koḍu (koṭṭ-) 'give' and Kota koṛ- 'give'; Census koṭɔ̄̃vtu is the give stem with infinitive morphology.")
add([250], 3582, "DEDR 3582 gives Tamil naṭa and Kannada naḍe 'walk'; Census naṭəvətu is a corresponding infinitive.")
add([262], 1097, "DEDR 1097 has Tamil kaccu and Kannada kaccu 'bite'; Census kaccɔ̄̃tu retains the bite stem.")
add([266], 1147, "DEDR 1147 lists Tamil kaṭṭu and Kannada kaṭṭu 'tie, fasten'; Census kaṭṭɔ̄tu is the infinitive of that stem.")
add([276], 2426, "DEDR 2426 gives Tamil cā, Kannada sāy and Kannada sāvu 'die/death'; Census sāvətu is a die infinitive.")
add([282], 1835, "DEDR 1835 lists Kannada kuḷḷu 'sit down' and Tulu kuḷḷuni 'sit'; Census kulibatu is the Badaga sit infinitive also supported by Badaga LSI kuli.")
add([286], 5350, "DEDR 5350 lists Kannada bāyu 'swell' and bāvu 'swelling'; Hockings' Badaga bāvu denotes a swelling or bulge.")
add([295], 2017, "DEDR 2017(a) has Tamil kēḷ and Kannada kēḷu 'hear, listen'; Census kɛ̄ppatu reflects the elicited hear verb with infinitive morphology.")
add([298], 707, "DEDR 707 lists Kota org- 'sleep' and Kannada oṟagu/oṟaṅgu 'lie down, sleep'; Census oṛəgɔ̄tu has the same sleep stem.")
add([354, 372], 3584, "DEDR 3584 lists Kannada naḍu/naḍuvu 'middle, centre' and naḍuve 'between'; both Census naḍuvə records have those respective senses.")
add([361], 5020, "DEDR 5020(a) gives Tamil muṉ 'before, in front' and Kannada munde 'before'; Census mundāṭɛ has the same front/before base with a local ending.")
add([380], 410, "DEDR 410(a) is the proximal i- demonstrative article and includes local 'here' forms; Hockings' Badaga illi 'here' is directly assigned to it.")
add([383], 5160, "DEDR 5160 has Tamil nāṉ and Kannada nān 'I'; Census nā is the reduced Badaga first-person form also recorded by Badaga LSI.")
add([385], 410, "DEDR 410(a) includes Tamil itu and Kannada idu 'this'; Census itu is the proximal neuter, although the elicitation gloss includes a dative sense not visible in the form.")
add([388, 389], 5160, "DEDR 5160 gives the first-person oblique en- alongside Tamil nāṉ and Kannada nān; Census eñəgə̄ 'to me' and ennutu 'my' use that oblique base.")
add([391], 2559, "DEDR 2559 has Tamil illai and Malayalam/Kannada illa 'not, does not exist'; Hockings' Badaga illa 'no, not' is the exact local counterpart.")
add([393], 1, "DEDR 1 is the distal a- demonstrative article and includes Tamil atu and Kannada adu 'that'; the separate Census atu 'that' record matches its distal sense.")
add([396], 1, "DEDR 1 includes distal place expressions such as Tamil aṅku 'there'; Hockings' Badaga alli 'there' directly matches Census aḷḷi.")
add([401, 403], 5151, "DEDR 5151 contains the interrogative e- series, including Malayalam ēn 'what' and Kannada elli 'where'; Census ēnə and eḷḷi match these question words, with Hockings elli 'where' explicit.")
add([408], 3688, "DEDR 3688 lists Kannada nīvu/nīṅgaḷ 'you plural' and the corresponding Badaga nīŋla; Census niŋgətu 'your' has the plural oblique stem.")
add([420], 4879, "DEDR 4879 lists Kannada mīse, Kota mi·c and Malayalam mīśa 'moustache'; Census mīsə is a close same-sense match.")

assert set(proposals) <= set(range(1, len(records) + 1))
accepted = []
for index, (parent, evidence) in sorted(proposals.items()):
    record = records[index - 1]
    ids = parent if isinstance(parent, list) else [parent]
    accepted.append({"record": record, "parent": f"d{ids[0]}",
                     "components": [f"d{x}" for x in ids] if len(ids) > 1 else None,
                     "kind": "component" if len(ids) > 1 else "reflex",
                     "citation": record["Source"] + ";" + ";".join(f"dedr[entry {x}]" for x in ids),
                     "evidence": evidence, "status": "pending-review", "surveyIndex": index})
for row in accepted:
    if row["components"] is None:
        del row["components"]

holds = []
for index, record in enumerate(records, 1):
    if index in proposals:
        continue
    form, gloss = record["Form"], record["Gloss"]
    if form.lower() in {"no data", "no vord"}:
        reason = "The source explicitly gives no usable Badaga lexical form for this elicitation."
    elif any(c in form for c in [" ", "[", "]"]):
        reason = f"The elicited response {form!r} is multiword, segmented uncertainly, or editorially bracketed; its components need separate verification before an etymon can be assigned."
    elif index in {2, 4, 10, 20, 30, 31, 44, 65, 97, 98, 122, 123, 125, 128, 133, 136, 142, 147, 148, 151, 152, 153, 154, 159, 161, 162, 166, 168, 169, 321, 324, 325, 332, 334, 338, 343, 349, 352, 409, 419, 421}:
        reason = f"{form!r} ‘{gloss}’ is loanlike or circulates in regional vocabulary; an attested immediate donor for this Badaga form has not been established."
    elif index in {74, 94, 106, 114, 209, 256, 259, 315, 343, 384, 398}:
        reason = f"The exact Census form {form!r} has a competing sense or apparent homonym in Badaga comparative sources; the source gloss ‘{gloss}’ needs reconciliation before linking."
    elif index in {57, 63, 77, 105, 116, 121, 174, 178, 180, 181, 188, 190, 214, 219, 237, 271, 273, 290, 294, 296, 301, 314, 327, 337, 363, 370, 371, 377, 381, 382, 390, 394, 399, 407, 418}:
        reason = f"{form!r} ‘{gloss}’ has a possible lexical-family comparison, but the article subsection, Badaga sound development, or inflected stem is not yet secure."
    else:
        reason = f"{form!r} ‘{gloss}’ remains held for a dialect-specific comparandum, immediate donor, or internal segmentation; homepage similarity alone is insufficient."
    holds.append({"record": record, "reason": reason, "status": "research-hold", "surveyIndex": index})

parents = {pid for row in accepted for pid in row.get("components", [row["parent"]])}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as stream:
    for row in csv.DictReader(stream):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, sorted(parents - set(articles))
assert len(accepted) + len(holds) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": holds}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(holds, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(len(accepted), "proposed records,", len(holds), "holds,", sum(len(row.get("components", [row["parent"]])) for row in accepted), "rows")
