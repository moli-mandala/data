"""Research-only Eravallan Varghese 2015 residual decisions."""
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

add([1, 2], 5031, "DEDR 5031 includes Tamil mūñci, Malayalam muññi/mōnta, and Kodagu mu·ḍï 'face'; survey mūñḍźi has the same nasal-retroflex/affricate face stem in both localities.")
add([3, 4], 1448, "DEDR 1448 lists Malayalam kātu and Irula kādu 'ear'; the two kādɨ source records match this ear noun.")
hold([5, 6], "nagam 'fingernail' resembles regional nakha/nakham contact vocabulary; immediate donor is unresolved.")
hold([7, 8], "idayam 'heart' resembles hṛdaya-type loan vocabulary; immediate donor is unresolved.")
add([9, 10], 3911, "DEDR 3911 directly gives Tamil pati 'town, city, village' and Malayalam pati 'fixed settlement'; both survey pati 'village' records preserve this settlement family.")
add([11, 13], 1904, "DEDR 1904 gives Malayalam kūra 'hut, thatch' and Tamil kūrai 'roof, small hut'; both survey kūra 'house' records have the same form with a broader shelter gloss.")
add([12], 5393, "DEDR 5393 gives Malayalam vīṭu and Tamil vīṭu 'house, abode'; survey vīḍɨ matches the house noun with survey voicing and final vowel.")
hold([14], "simarɨ 'broom' has no secure first-hand lexical match in the checked DEDR families.")
add([15, 19], [184, 1298], "Ordered grinding-stone + stone: DEDR 184 gives Malayalam ammi, a grinding stone, and 1298 gives Malayalam kal/kallu 'stone'. The source's pestle gloss for the second locality may be a local tool extension and is retained.")
add([16], [4867, 228, 1298], "Ordered pepper/chilli + grind + stone: DEDR 4867 gives Malayalam muḷaku 'pepper', 228 Malayalam arayuka/arekkuka 'grind', and 1298 kall 'stone'. The whole moḷakarakallɨ response names a spice-grinding stone.")
add([17], [228, 1298], "Ordered grind + stone: DEDR 228 gives Malayalam arayuka/arekkuka 'grind' and 1298 gives kall 'stone'; the survey preserves both words in ara kallɨ.")
add([18], 1819, "DEDR 1819 gives Tamil kur̤avi 'grinding pestle' and Malayalam kur̤avi 'small rolling stone to grind with'; survey koḷavi matches the tool and final -avi, with a different liquid consonant.")
add([20, 21], 2063, "DEDR 2063 explicitly lists Malayalam koṭṭi 'mallet' beside Tamil koṭṭu 'hammer/beat'; both survey koṭṭi 'hammer' forms match directly.")
hold([22, 23], "koḍāli 'axe' has DEDR Appendix 32 and CDIAL 3244 comparisons; its immediate regional loan donor and direction remain unresolved.")
hold([24, 25], "modiram 'ring' resembles a widespread Indo-Aryan loan; immediate regional donor unresolved.")
hold([26, 27], "sūriya 'sun' is sūrya-type learned/contact vocabulary; immediate donor unresolved.")
add([28, 29], 3045, "DEDR 3045 lists Malayalam and Tamil taṇṇi 'water'; both survey ta ṇṇi records preserve the source spacing and refer to this water noun.")
hold([30], "cāram 'ash' does not securely match DEDR 2453 cāmpal without unexplained loss of the nasal-stop cluster.")
add([31], 3283, "DEDR 3283 gives Malayalam tūḷ and Tamil tūḷ 'dust'; survey spaced to ḷam shares the t-ḷ dust stem with a changed vowel and ending.")
add([32], 4782, "DEDR 4782 directly gives Tamil māṅkāy 'unripe mango' and Malayalam māṅṅa 'mango fruit'; survey māŋgakāi preserves the full mango-fruit formation.")
hold([33, 34], "godimbi 'wheat' is a regional crop/contact term with no verified immediate donor.")
add([35, 36], 4867, "DEDR 4867 includes Malayalam muḷaku 'pepper', Kota meḷg 'chilli', and Kodagu maḷu 'chillies'; both mɔḷagɨ records show the pepper-to-chilli sense in this family.")
hold([37], "veḷḷɛ pūṇḍɨ 'garlic' appears to contain 'white' plus a plant name, but the second element's exact lexical and contact history is unresolved.")
hold([38, 39], "vaŋgaya/vɛŋkayam 'onion' are regional crop forms with uncertain immediate donor and internal composition.")
hold([40, 41], "ta kkāḷi 'tomato' is a post-Columbian crop term with uncertain regional borrowing path.")
hold([42, 43], "mɔṭakosɨ 'cabbage' is a regional crop term with unresolved immediate donor.")
hold([44], "vaṇṇatti 'spider' is not the DEDR 5241 vaṇṇātti 'washerwoman' homonym; no spider etymology is secure.")
hold([45, 46], "ravuḷan 'husband' lacks a secure DEDR form-and-sense match; resemblance to āḷan is insufficient.")
hold([47, 48], "raviṭi 'wife' may be locally paired with ravuḷan but the morphology and historical parent are unresolved.")
hold([49, 50], "ambuḷapayyan 'boy' is multi-part, probably including payyan, but its first/community element is not securely analysed.")
hold([51], "jamam 'night' appears to be yāma-type time vocabulary; immediate donor unresolved.")
hold([52], "rātri 'night' is rātri-type contact vocabulary; immediate donor unresolved.")
hold([53, 54], "madyanam 'noon' is madhyāhna-type borrowed time vocabulary; immediate donor unresolved.")
hold([55], "poḷuto ḍɨ 'evening/afternoon' appears to include a time/day element, but the whole spaced response needs ordered analysis.")
hold([56, 57], "vaineram 'evening/afternoon' appears compound, with possible nēram 'time'; the first constituent is unresolved.")
hold([58, 59], "vāra m 'week' is regional calendar vocabulary with an unresolved immediate donor.")
hold([60, 61, 62, 63], "All four source-defined māsam 'month' records retain distinct IDs; the likely māsa-type immediate loan donor is unresolved.")
hold([64, 65, 66, 67], "All four source-defined varśam 'year' records retain distinct IDs; the varṣa-type immediate donor is unresolved.")
hold([68, 69], "mośam/mosam 'bad' is a regional contact term; its immediate donor and semantic path are unresolved.")
add([70], 1670, "DEDR 1670 gives Malayalam kuṭu 'small, narrow' and Toda kuṭ 'short, small'; survey kuṭa 'short' matches this smallness family.")
hold([71, 72], "viravirappɨ 'cold' appears expressive or derived; no secure parent and derivational structure have been identified.")
hold([73], "dūram 'far' resembles Indo-Aryan dūra; immediate donor unresolved.")
add([74, 75], 5496, "DEDR 5496(a) gives Malayalam veḷḷa 'white colour'; both survey vɛḷḷa records preserve the white adjective.")
add([76, 77], 5052, "DEDR 5052 gives Malayalam mūnnu, Kota mu·nḏ and Kodagu mu·ndï 'three'; survey mūntɨ matches the nasal-stop numeral family.")
add([78, 79], [3918, 990], "Ordered ten + one: DEDR 3918 gives Tamil patiṉ- and Malayalam patin- 'ten' in teen numerals; 990 gives the or-/on- one family. Both padinontɨ eleven records preserve the compound.")
add([80, 81], 5151, "DEDR 5151 explicitly gives Malayalam/Tamil ār 'who'; both source ārɨ records add a final vowel.")
add([82], 5151, "DEDR 5151 gives Tamil eṉṉa 'what' and Malayalam entu; survey ɛnnā retains the e-n-n interrogative stem.")
hold([83, 84], "ɛndɨkoḷɨ 'what?' is a complex response with an interrogative and a possible koḷ element; do not collapse it to one root.")
add([85, 86], 5151, "DEDR 5151 gives Malayalam eṅṅu and Tamil eṅku/eṅkē 'where'; both eŋgɨ responses retain the interrogative place stem.")
add([87, 88], 5151, "DEDR 5151 contains Malayalam eppōḷ and Tamil eppōtu 'when'; both survey ɛppa responses preserve that temporal interrogative base.")
add([89, 90], 5151, "DEDR 5151 lists Malayalam ettira/etra and Tamil ettaṉai 'how much, how many'; the two locality forms ɛttra and ɛtɨna fit the same quantity-interrogative family.")
add([91, 92], 410, "DEDR 410(a) explicitly lists Malayalam itu and Kota id 'this'; survey itɨ/idɨ are proximal neuter demonstratives.")
add([93, 94], 1, "DEDR 1 lists Malayalam atu and Kota ad 'that'; both adɨ records are distal neuter demonstratives.")
hold([95], "orɨtiyatrikɨ 'same' appears multi-part, but its intervening and final morphemes remain unresolved.")
add([96], [990, 4597], "Ordered one + like: DEDR 990 gives Malayalam oru/ōr 'one, the same', and 4597 gives Malayalam pōla 'like'; ontɨpola expresses sameness as 'one-like'.")
hold([97], "pūra m 'whole' resembles pūrṇa/pūrā contact vocabulary; immediate donor unresolved.")
add([98, 99], 946, "DEDR 946 gives Malayalam oṭiyuka/uṭayuka 'break' and Kodagu oḍe- 'be broken'; survey ḍɛnjɨtɨ is a reduced inflected broken form in that family, with initial-vowel loss requiring review.")
hold([100, 101], "orupaḍɨ 'many' contains apparent oru 'one' but the remaining measure/quantity semantics are unclear; do not flatten it.")
hold([102], "sappiḍɨ 'eat' is a regional cāppiṭu-type verb not securely placed in DEDR; contact history uncertain.")
add([103, 104], 1124, "DEDR 1124 gives Malayalam kaṭikka and Kannada/Kodagu kaḍi 'bite'; both survey kaḍi forms match the bite stem despite source wording 'he bite'.")
add([105], 3825, "DEDR 3825 gives Tamil paci 'hunger, be hungry' and Malayalam payikka 'be hungry'; survey paśikatɨ retains the pac-/pas- hunger stem with ordinary predicative morphology.")
hold([106, 107], "vayarɨkavatɨ 'he is hungry/was hungry' appears to contain 'belly' and another predicate; ordered components are unresolved.")
hold([108, 109], "ta vamakɨtɨ 'thirsty/he was thirsty' is a spaced predicate whose lexical stem and auxiliary are unresolved.")
hold([110], "ukkarɨ 'sit down' has no secure DEDR parent; a possible contact verb needs an immediate donor.")
hold([111, 112], "nokkɨrɨ is glossed 'sit down' although DEDR 3794 nōkk- means 'look'; the source-form/gloss conflict remains unresolved.")
add([113, 114], 811, "DEDR 811 gives Malayalam eriyuka and Tamil eri 'burn'; both ɛrikitɨ forms retain the burn stem with survey tense morphology.")
hold([115, 116], "cakaṇḍa occurs under the contradictory 'don't die, he died' prompt; negative/past and stem segmentation require review.")
hold([117], "kollarutɨ combines the kill stem with a negative/prohibitive element in a contradictory elicitation prompt; no single-root assignment.")
hold([118], "kontɨkaḷayarutɨ is a multi-part kill/prohibition response requiring component analysis.")
add([119, 120], 4572, "DEDR 4572 gives Malayalam pōka and Tamil pō 'go'; poyaṭɨ retains the go stem in the elicited past form.")
add([121], 4091, "DEDR 4091(a) gives Tamil pār 'look' and Malayalam pārkka 'behold'; survey parɨ is the corresponding look stem.")
add([122], 3794, "DEDR 3794 explicitly gives Malayalam nōkkuka and Tamil nōkku 'look, see'; survey nokkɨ is a direct look response, distinct from the mismatched sit-gloss nokkɨrɨ records.")
add([123, 124], 5160, "DEDR 5160 gives Tamil nāṉ 'I' and Malayalam ñān; the two nānɨ survey forms retain the Tamil-shaped first-person singular pronoun.")
add([125, 126, 127], 3684, "DEDR 3684 gives Tamil/Malayalam nī 'you (sg.)'; the source's two informal and one formal ni records have the same singular stem, with formality a local pragmatic distinction.")
add([128], 3688, "DEDR 3688 gives Malayalam niṅṅaḷ 'you (pl.)', also usable for respectful address; nīŋga is the formal singular locality response.")
add([129, 130], 1, "DEDR 1 explicitly lists Malayalam avan 'that man, he'; both survey avan forms match the masculine distal pronoun.")
add([131, 132], 1, "DEDR 1 gives Malayalam avaḷ 'that woman, she'; survey ava is a shorter feminine distal form in both localities.")
add([133, 134], 3647, "DEDR 3647 gives Malayalam nām (oblique nam-) 'we, I and you'; the two nammɨ forms are explicitly inclusive.")
add([135, 136], 5154, "DEDR 5154 gives Malayalam ñāṅṅaḷ 'we' exclusive and Kodagu naŋga; both survey naŋkaḷ/naŋga forms are explicitly exclusive.")
add([137, 138], 3688, "DEDR 3688 gives Malayalam niṅṅaḷ 'you (pl.)'; the two niŋgaḷ source records are plural.")
hold([139, 140], "aikkaḷɨ/aikka 'they' might contract a distal plural but neither the stem nor the final morphology matches DEDR 1 securely.")

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
(P / "record-coverage-001.json").write_text(json.dumps({"source": "varghese2015palakkad",
    "language": "Eravallan", "residualSourceRecords": len(records), "proposedRecords": len(accepted),
    "heldRecords": len(held), "unexaminedRecords": 0, "uniqueOwnership": True,
    "recordOwnership": owners}, ensure_ascii=False, indent=2) + "\n")
print(f"{len(accepted)} proposed records, {len(held)} holds, {sum(len(x.get('components', [x['parent']])) for x in accepted)} rows")
