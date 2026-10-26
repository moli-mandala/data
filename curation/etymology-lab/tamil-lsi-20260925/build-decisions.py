"""Draft Tamil LSI research decisions only; never writes accepted sidecars."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 148
proposals = {}


def assign(indices, parents, evidence, kind="reflex"):
    if isinstance(parents, str):
        parents = [parents]
    for i in indices:
        assert i not in proposals
        proposals[i] = (parents, evidence, "component" if len(parents) > 1 else kind)


assign([3], "d784", "DEDR 784 lists Tamil eṭṭu ‘eight’ and eṇ- in eṇ-patu ‘eighty’; LSI eṇ is the numeral combining stem, not the homophonous ‘count’ of DEDR 793.")
assign([4, 5], "d474", "DEDR 474 explicitly lists Tamil iru before consonants and īr before vowels as ‘two’; both LSI forms are numeral allomorphs.")
assign([6, 7, 8], "d5052", "DEDR 5052 lists Tamil mūṉṟu ‘three’ and adjectival mu-/mūv-; the LSI variants represent this numeral family.")
assign([18], "d3085", "DEDR 3085 directly gives Tamil tampi ‘younger brother or male parallel cousin’; the survey's broad ‘Brother’ gloss lacks that age distinction.")
assign([19], "d23", "DEDR 23 explicitly lists Tamil akkāḷ among the elder-sister and female-kin forms.")
assign([20], "d3015", "DEDR 3015(a) directly lists Tamil taṅkacci ‘younger sister or female parallel cousin’; LSI taŋgaccĭ matches this form.")
assign([27, 29, 138], "d1711", "DEDR 1711(a) directly gives Tamil kutirai ‘horse’ and South Dravidian cognates; the Indo-Aryan caveat concerns the separate Telugu guṟṟamu branch (b). LSI -gaḷ is ordinary plural inflection.")
assign([28, 30], ["d4395", "d1711"], "Tamil peṇ ‘female’ is in DEDR 4395(a) and kutirai ‘horse’ in DEDR 1711(a); LSI ‘mare(s)’ preserves these two ordered components, with plural -gaḷ unanalysed as inflection.")
assign([31, 33], "d4798", "DEDR 4798 explicitly gives Tamil māṭu ‘ox’; LSI ‘Bull(s)’ narrows that cattle sense, and -gaḷ is ordinary plural inflection.")
assign([35, 37], ["d4395", "d3650"], "DEDR 4395(a) gives Tamil peṇ ‘female (of animals)’ and DEDR 3650 gives Tamil nāy ‘dog’; the ordered expression denotes female dog(s).")
assign([36], "d3650", "DEDR 3650 directly gives Tamil nāy ‘dog’; LSI -gaḷ is regular plural inflection.")
assign([38], ["d4395", "d5152"], "DEDR 4395(a) gives Tamil peṇ ‘female’ and DEDR 5152 gives Tamil āṭu ‘goat’; LSI ‘Female goat’ combines those forms in order.")
assign([39], "d5152", "DEDR 5152 directly gives Tamil āṭu ‘goat, sheep’; LSI -gaḷ is regular plural inflection.")
assign([40], ["d399", "d4780"], "DEDR 399 gives Tamil āṇ ‘male’ and DEDR 4780 gives māṉ ‘deer’; the LSI expression contains both ordered components.")
assign([41], ["d4395", "d4780"], "DEDR 4395(a) gives Tamil peṇ ‘female’ and DEDR 4780 gives māṉ ‘deer’; the LSI expression contains both ordered components.")
assign([42], "d4780", "DEDR 4780 explicitly gives Tamil māṉ ‘deer’, beside other Dravidian deer words.")
assign([43], ["d3610", "d1"], "DEDR 3610 gives Tamil nalla ‘good’ and DEDR 1 gives avaṉ ‘he, that man’; LSI nallavan is the good-man expression with ordinary sandhi.")
assign([47], ["d1942", "d3939"], "DEDR 1942 gives Tamil keṭṭa ‘bad’ and DEDR 3939 gives paiyaṉ ‘boy’; LSI peiyan denotes their ordered adjective–noun expression.")
assign([48], ["d1942", "d4395", "d4198"], "DEDR 1942 gives Tamil keṭṭa ‘bad’, DEDR 4395 peṇ ‘female’, and DEDR 4198 piḷḷai ‘child’; the source has all three ordered words.")
assign([49], "d646", "DEDR 646 gives Tamil uyar ‘be high’ and uyarn(t)- inflected forms; DEDR 647 is the homophonous ‘disappear’ family and is excluded.")
assign([50], "d5086", "DEDR 5086 directly gives Tamil mēl ‘over, above’ and mēlē as a locative/adverbial form.")
assign([51], "d1619", "DEDR 1619 gives Tamil kīḻ ‘below, down’; LSI kīṛē is its adverbial form with source transcription of the retroflex lateral.")
assign([52], "d222", "DEDR 222 gives Tamil aruku ‘nearness, vicinity’ and the Malayalam arikē ‘near’; LSI arugē preserves the same local expression.")
assign([54], "d5020", "DEDR 5020(a) directly gives Tamil muṉ ‘in front, before’ and muṉpu ‘before’.")
assign([55], "d4205", "DEDR 4205 directly gives Tamil piṉpu ‘afterwards, posterior in place or time’; LSI pinpŭ has the same sense.")
assign([56, 57, 58], "d5151", "DEDR 5151 gives Tamil yār ‘who’, interrogative eṉṉa ‘what’, and ēṉ ‘why’ within its full interrogative family.")
assign([61], "d333", "DEDR 333 explicitly gives Tamil ām ‘yes’, expressing assent, from the āku ‘become’ paradigm.")
assign([62], "d2559", "DEDR 2559 explicitly gives Tamil illai ‘it is not; no’; LSI illei is the survey spelling.")
assign([63], "d4572", "DEDR 4572 gives Tamil pō ‘go’, including its pōv-/pōkuv- paradigm.")
assign([65], "d474", "DEDR 474 explicitly gives Tamil iru-patu ‘twenty’ as the ‘two’ numeral compound; the complete numeral is attested in that article.")
assign([66], "d2826", "DEDR 2826 explicitly gives Tamil aim-patu/am-patu ‘fifty’; LSI eimbadu retains the five-plus-ten compound with survey notation.")
assign([67], "d5160", "DEDR 5160 lists the Tamil first-person oblique eṉ- ‘my’ beside nāṉ ‘I’.")
assign([68, 70], "d5154", "DEDR 5154 explicitly lists Tamil nāṅkaḷ ‘we (exclusive)’ with oblique eṅkaḷ-; the LSI subject and possessive forms stay in this paradigm.")
assign([69], "d3647", "DEDR 3647 gives Tamil nām ‘we’ with oblique nam(m)-; LSI nam ‘Our’ belongs to this inclusive paradigm, separate from exclusive eṅkaḷ.")
assign([71, 73, 74, 120], "d3688", "DEDR 3688 explicitly gives Tamil nīr (plural or honorific singular), nīṅkaḷ ‘you’, and oblique um-/uṅkaḷ-; the source uses both address and possessive forms.")
assign([72], "d3684", "DEDR 3684 gives Tamil nī ‘thou’ with modern oblique uṉ- ‘thy’; this is singular, distinct from DEDR 3688's um-.")
assign([75], "d5270", "DEDR 5270 explicitly gives Tamil varu ‘come’ with imperative vā and past vant-; LSI vā is the imperative.")
assign([76], "d1041", "DEDR 1041 directly gives Tamil ōṭu ‘run’; LSI ōḍ is the same verbal base in survey transcription.")
assign([89], "d77", "DEDR 77 gives Tamil aṭi ‘beat, strike’ with -pp-/-tt- verbal stems; the bare elicited verb belongs to this family.")
assign([121, 122, 123, 124], "d1", "DEDR 1 explicitly gives Tamil avaṉ ‘that man, he’ and avar ‘those persons, honorific person’; the LSI ‘his/their’ glosses are possessive uses of those pronouns.")
assign([126], "d1159", "DEDR 1159(a) gives Tamil kaṇ ‘eye’; LSI kaṇa has a survey-final vowel but the same stem and meaning.")
assign([127], "d1448", "DEDR 1448 directly gives Tamil kātu ‘ear’; LSI kādu reflects survey voicing notation.")
assign([128], "d5259", "DEDR 5259 directly gives Tamil vayiṟu ‘belly’; LSI vayarŭ is its survey transcription.")
assign([129], "d4955", "DEDR 4955 directly gives Tamil mutuku ‘back, spine region’; LSI mudugŭ follows the same form with survey voicing notation.")
assign([130], "d486", "DEDR 486 directly gives Tamil irumpu ‘iron’, also Malayalam and Kodagu comparanda; LSI irumbŭ is its survey transcription.")
assign([131], "d72", "DEDR 72 directly gives Tamil aṭimai ‘slavery, slave’ alongside aṭi ‘foot’; LSI aḍimei is that derivative within the article.")
assign([133], "d450", "DEDR 450 gives Tamil iṭaiyar ‘herdsmen’ and Malayalam iṭayan ‘shepherd’; LSI iḍeiyan is a singular form of the same herdsman term.")
assign([134], "d5093", "DEDR 5093 explicitly lists Tamil mēyppaṉ ‘herdsman, shepherd’ from mēy ‘graze’.")
assign([135], "d1109", "DEDR 1109 explicitly lists Tamil kaṭavuḷ ‘god’ within the kaṭa ‘cross, transcend’ article; the displayed article head differs from this attested member.")
assign([137], "d5393", "DEDR 5393 explicitly gives Tamil vīṭu ‘house, habitation’ within its wider viṭu ‘release’ article; LSI vīḍŭ matches that sense.")
assign([139], "d1364", "DEDR 1364 directly gives Tamil kaḻutai ‘ass’ and related Dravidian forms; LSI kaṛudei records the retroflex lateral with ṛ.")
assign([142], "d4355", "DEDR 4355 explicitly lists Tamil pūṉai ‘cat’; LSI pūnei has a survey diphthong spelling.")
assign([143], "d4020", "DEDR 4020 directly gives Tamil paṟavai ‘bird’ within its paṟa ‘fly’ family; LSI paṛavei is the same member.")
assign([144], "d2818", "DEDR 2818 directly gives Tamil cēval ‘cock’; LSI śēval reflects survey spelling.")
assign([145], ["d4395", "d3169"], "DEDR 4395(a) gives Tamil peṇ ‘female’ and DEDR 3169 gives Tamil tārā ‘duck’; LSI ‘Duck’ specifies a female duck in its form.")

articles = {}
ids = {p for parents, _, _ in proposals.values() for p in parents}
with gzip.open(ROOT / "data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in ids:
            articles[row["Entry_ID"]] = {"Page": row["Page"], "Citation": row["Citation"], "Raw_Article": row["Raw_Article"], "URL": f"https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={row['Page']}"}
assert ids == set(articles), ids - set(articles)
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")

hold_reasons = {}
for i in [1, 2]: hold_reasons[i] = "The source numeral spelling is anomalous against the directly attested DEDR forms; do not normalize it by assumption."
for i in range(9, 18): hold_reasons[i] = "The tagappanār kin term or its multiword/case-marked expression lacks a securely identified immediate etymon; no component-only link to ‘father’ is assigned."
for i in [21, 22, 24, 25, 26, 32, 34, 53, 125, 136, 140, 141, 146, 147, 148]: hold_reasons[i] = "Likely or possible contact vocabulary; its immediate donor is not established by a verified persistent donor node."
for i in [23]: hold_reasons[i] = "The peṇḍadĭ wife form is not directly listed as Tamil in DEDR 4395, though Kannada peṇḍati is; immediate source unresolved."
for i in [44, 46]: hold_reasons[i] = "The multiword expression includes manidar or istirĭ, whose immediate source is unresolved; linking only the adjective would be incomplete."
for i in [45]: hold_reasons[i] = "LSI mallvaḷ ‘Good woman’ may be a transcription error for nallavaḷ; the first segment cannot be silently normalized."
for i in [59, 60]: hold_reasons[i] = "The conjunction/discourse expression needs its own grammatical or lexical source analysis; a same-spelling dictionary hit is insufficient."
for i in [64]: hold_reasons[i] = "Uṭkār ‘sit’ has no secure direct DEDR match in the checked material; do not infer a root from surface similarity."
for i in list(range(77, 89)) + list(range(90, 120)):
    hold_reasons[i] = "A full subject-plus-predicate clause with person, tense and agreement morphology; the verb root alone is not the etymology of the whole source response."
for i in [132]: hold_reasons[i] = "Payiriḍugiṛavan ‘cultivator’ is a multi-morpheme participial expression requiring complete component analysis."

accepted = []
held = []
for i, rec in enumerate(records, 1):
    if i in proposals:
        parents, evidence, kind = proposals[i]
        citation = rec["Source"] + ";" + ";".join(f"dedr[{p[1:]}]" for p in parents)
        d = {"record": rec, "parent": parents[0], "kind": kind, "citation": citation,
             "evidence": evidence, "status": "pending-review", "surveyIndex": i}
        if len(parents) > 1: d["components"] = parents
        accepted.append(d)
    else:
        assert i in hold_reasons, (i, rec["Form"])
        held.append({"record": rec, "reason": hold_reasons[i], "status": "research-hold", "surveyIndex": i})
assert len(accepted) + len(held) == 148
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
(P / "record-coverage-001.json").write_text(json.dumps({"residualSourceRecords": 148, "proposedRecords": len(accepted), "heldRecords": len(held), "unexaminedRecords": 0, "uniqueOwnership": len({x["record"]["ID"] for x in accepted + held}) == 148}, indent=2) + "\n")
print(json.dumps({"proposedRecords": len(accepted), "assignmentRows": sum(len(x.get("components", [x["parent"]])) for x in accepted), "heldRecords": len(held), "unexamined": 0, "articles": len(articles)}))
