"""Conservative, article-backed proposals for residual Irula Varghese records."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 643
proposed = {}
holds = {}

def add(indices, number, comparison):
    for i in indices:
        assert i not in proposed, i
        proposed[i] = (number, comparison)

def hold(indices, reason):
    for i in indices:
        assert i not in holds, i
        holds[i] = reason

# Full DEDR articles and existing Irula/Tamil comparanda support the selected proposals.

add([3, 4], 5031, "Tamil mūñci 'face' and Kodagu mūḍɨ 'face' match Irula mūñḍźi.")
add(list(range(5, 14)), 1448, "DEDR records Irula kādu and Tamil kātu 'ear', matching the kātɨ/kādɨ village variants.")
add([14], 3633, "Kannada nālage and Tulu nālayi 'tongue' match Irula nāḷāŋge.")
add([15], 2023, "Kannada/Tulu kai/kayi 'hand, arm' matches Irula kɛya 'arm'.")
add([21], 1479, "Tamil/Kannada kāl and Kodagu kālɨ 'leg' match Irula kælɨ.")
add([30], 3911, "Tamil pati 'town, city, village' matches Irula pati 'village' in the full settle/abode article.")
add([31, 32, 33, 34, 35, 36, 37, 39, 40], 1904, "Malayalam kūra 'hut, thatch' matches Irula kūrɛ/kūra/kure 'house'.")
add([38], 5393, "Tamil vīṭu 'house' matches Irula vīḍɨ within the full article.")
add([60, 61], 2063, "Malayalam koṭṭi 'mallet' and Tamil koṭṭu 'hammer, beat' match Irula koṭṭi 'hammer'.")
add([85, 87, 88], 4559, "The time/sun article includes Kannada/Tulu pottu/portu and Konda podu 'sun', matching Irula podu.")
add(list(range(97, 105)), 3045, "Tamil/Malayalam taṇṇi 'water' matches the Irula ta ṇṇi responses, despite spacing in the transcription.")
add([131, 134], 497, "Tamil ilai and Tulu irɛ 'leaf' match Irula ilɛ/lɛyi in the full leaf article.")
add([132, 133], 2673, "Malayalam cappu and Tulu coppu 'leaf' match Irula coppa.")
add([186], 4096, "Tamil/Malayalam pāl 'milk' matches Irula pælɨ.")
add([206], 196, "Malayalam tamayan 'elder brother' matches Irula ta mayan, written with a space in the survey.")
add([211, 212, 213], 4616, "Tamil makaṉ/makaḷ and Kannada maga 'child, son' match Irula magæ/makæ 'son' within the full article.")
add(list(range(329, 338)), 5496, "Tamil veḷḷai and Malayalam veḷḷa 'white' match Irula vɛḷḷɛ/vɛḷḷa/vɛḷḷai.")
add([344], 474, "DEDR records Irula raṇḍu 'two', matching ra ṇḍɛ despite source spacing.")
add([345, 346], 5052, "Tamil mūṇu and Tulu mūji 'three' match Irula mūṇɛ/mūntu.")
add([347], 3655, "Tamil nāl 'four' matches Irula nālɛ 'four'; same-form 'tomorrow' is a distinct article and not used here.")
add([348], 2826, "Tamil añcu and Kodagu añji 'five' match Irula anjɛ.")
add([349], 2485, "Tamil āṟu 'six' matches Irula ārɛ.")
add([350], 910, "Kodagu ə̄ḷɨ 'seven' matches Irula āḷɨ 'seven'; same-form 'husband' in older pending work is a separate homograph.")
add([351], 784, "Tamil eṭṭu 'eight' matches Irula ɛṭṭɛ.")
add([352], 3918, "Tamil pattu 'ten' matches Irula pattɛ.")
add(list(range(362, 371)), 5151, "Kodagu ārɨ and Tamil yār 'who' match Irula ārɨ; same-form 'six' in older pending work is a separate homograph.")
add(list(range(371, 388)), 5151, "The full interrogative article includes Tamil eṉṉa/eṉṉatu 'what', matching Irula ɛnnātɨ/ɛnadɨ/ɛnna.")
add(list(range(388, 397)), 5151, "The full interrogative article includes Tamil eṅkē 'where', matching Irula ɛ̄ŋgɛ/ɛŋkɨ.")
add(list(range(397, 406)), 5151, "The full interrogative article includes Tamil eppōtu 'when', matching Irula ɛppo/ɛppā/ɛppa.")
add(list(range(406, 415)), 5151, "The full interrogative article includes Tamil ettaṉai 'how many', matching Irula ɛttɛn/ɛtta ni variants.")
add([415, 416, 418, 419, 422], 5151, "Tamil eṉṉa 'what kind' matches Irula ɛnna/ɛna/ɛnana in the interrogative family; oil homographs are excluded.")
add(list(range(424, 433)), 410, "Tamil itu and Kannada idu 'this' match Irula itɨ/idɨ.")
add(list(range(433, 442)), 1, "Tamil atu and Kannada adu 'that' match Irula atɨ/adɨ; goat homographs are excluded.")
add([479, 480, 481, 482, 483, 485, 486], 1124, "Tamil/Kannada kaṭi 'bite' matches Irula kaḍi.")
add([512, 513, 514, 515, 517, 518, 519, 520], 811, "Tamil eri and Malayalam eriyuka 'burn' match Irula ɛrɨkɨtɨ/ɛrikitɨ with local finite morphology.")
add([544, 545, 546], 1041, "Tamil ōṭu and Kannada ōḍu 'run' match Irula ōḍitta with past morphology.")
add([547, 548, 549, 550], 2017, "Tamil kēḷ 'hear, listen' matches Irula kēṭukɔ/kɛ̄kkaṇam in the full hearing article.")
add(list(range(551, 560)), 4091, "Tamil pār 'see, look' matches Irula parɨ/pærɨ/pǣrɨ 'look, he saw'.")
add(list(range(560, 569)), 5160, "Tamil nāṉ and Kannada nān 'I' match Irula nānɨ/nān.")
add(list(range(569, 578)), 3684, "Tamil/Kannada nī 'you singular' matches Irula ni/nī informal forms.")
add([578, 580, 581, 582, 585], 3684, "The bare ni response matches the singular second-person article despite the elicitation label 'formal'.")
add([579, 583, 584, 586], 3688, "Tamil nīyir/nīvir and related honorific-plural forms match Irula nīyɨ/niviri/nivarɨ/nimɨ.")
add(list(range(587, 605)), 1, "The distal demonstrative article includes Tamil avaṉ/avaḷ and Kannada avanu/avaḷu 'he/she', matching Irula avɛ/ava/avaḷɨ.")
add(list(range(605, 614)), 3647, "Tamil nām and Kannada nāvu 'we inclusive' match Irula namɨ/nām/nammɨ.")
add([620], 5154, "Tamil yām and related em-/eŋ- 'we exclusive' match Irula ɛmma.")
add([626, 627, 628, 629, 631, 632, 633, 634], 3688, "Tamil nīvir and related second-plural forms match Irula niviri/niviru/nivarɨ/nimɨ.")
add(list(range(635, 644)), 1, "Tamil/Kannada avar 'they' matches Irula avarɨ/avɛ̄rɨ in the distal demonstrative article.")

hold([2], "maṇḍe is glossed 'hair' here, but the direct DEDR 4682 comparison means 'head'; the specific source sense needs review.")
hold([16, 17, 18, 19, 20], "nāga/nagam 'fingernail' may reflect an Indo-Aryan nakha loan; the immediate source into Irula is unresolved.")
hold([26, 27, 28, 29], "īral/irudayam/ira kula are glossed 'heart' but involve liver or likely Sanskritic terms; sense and donor paths need review.")
hold(list(range(43, 60)), "The mortar/pestle responses are compounds ending in kallɨ 'stone'; the first component differs and needs individual whole-form analysis.")
hold(list(range(62, 71)), "koḍāli 'axe' may be a regional loan; the immediate donor into Irula is not established.")
hold(list(range(73, 82)), "modaram/modiram 'ring' resembles the Sanskrit mudrā family; immediate donor and adaptation remain unresolved.")
hold([82, 83, 84, 86, 89, 90, 91, 92, 93, 94, 95, 96], "The sun/moon/sky response is Sanskrit-shaped (sūrya, candra, ākāśa) or lacks a secure Dravidian article; immediate donor is unresolved.")
hold([106, 107, 109], "vani is glossed 'river' here but the same Varghese source form has older pending 'sky' links to DEDR 5381; source sense conflict.")
hold(list(range(113, 120)), "bimavalɨ 'rainbow' is morphologically opaque in this source; no secure whole-form article match.")
hold(list(range(135, 186)), "The crop/vegetable response belongs to recent contact vocabulary or a compound; immediate donor and whole-form analysis are unresolved.")
hold(list(range(237, 285)), "The week/month/year/bad responses are Sanskrit- or Indo-Aryan-shaped; immediate donor into Irula is unresolved.")
hold([290, 298, 299, 300, 301, 302, 303, 304, 305], "The response is a multiword or morphologically complex elicitation; the full construction needs analysis.")
hold(list(range(353, 362)), "The eleven numeral is an overt ten-plus-one formation; ordered component analysis is deferred.")
hold([417, 420, 421, 423], "The 'what kind' response is a different or multiword form; its complete morphology is unresolved.")
hold(list(range(442, 456)), "The demonstrative/sameness/difference response has overt plural or comparison morphology; whole-form analysis is unresolved.")
hold(list(range(458, 475)), "The broken/many response contains auxiliary or uncertain quantifier morphology; no secure whole-form analysis yet.")
hold(list(range(487, 507)), "The hunger/thirst response is a full phrase with verbal inflection; a single root link would not account for the whole response.")
hold(list(range(522, 540)), "The source elicitation combines negative and past clauses ('don't ...; he ...'), making the whole recorded response unclear.")
hold(list(range(614, 620)), "Exclusive 'we' is nearly identical to the source's inclusive namɨ/nāmu paradigm; clusivity needs independent review.")
hold(list(range(621, 626)), "This source labels the namɨ form exclusive despite matching its inclusive paradigm; clusivity requires review.")

for i, record in enumerate(records, 1):
    if i not in proposed and i not in holds:
        holds[i] = (f"{record['Form']} ‘{record['Gloss']}’: no secure whole-form parent follows "
                    "from the exact-form searches and checked DEDR articles; retain for further analysis.")
assert not set(proposed) & set(holds)
assert set(proposed) | set(holds) == set(range(1, len(records) + 1))

accepted = []
research_holds = []
for i, record in enumerate(records, 1):
    if i in proposed:
        number, comparison = proposed[i]
        accepted.append(dict(record=record, parent=f"d{number}", kind="reflex",
                             citation=record["Source"] + f";dedr[entry {number}]",
                             evidence=f"DEDR {number} gives {comparison}", status="pending-review",
                             surveyIndex=i))
    else:
        research_holds.append(dict(record=record, reason=holds[i],
                                   status="research-hold", surveyIndex=i))
parents = {item["parent"] for item in accepted}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in parents:
            # The audit cache has a second, conflicting 5282 label on p. 487;
            # the compiled d5282 node cites the first p. 477 vali/pull article.
            if row["Entry_ID"] in articles:
                assert row["Entry_ID"] == "d5282" and articles[row["Entry_ID"]]["Page"] == "477"
                continue
            articles[row["Entry_ID"]] = {k: row[k] for k in ("Page", "Citation", "Raw_Article")}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, parents - set(articles)
(P / "proposed-decisions.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted,
                                                            held=research_holds), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(research_holds, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(dict(proposed=len(accepted), rows=len(accepted), held=len(research_holds), parents=len(parents)))
