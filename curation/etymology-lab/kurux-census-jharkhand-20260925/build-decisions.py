"""Conservative, article-backed proposals for residual Kurux Jharkhand Census records."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 403
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

# The compared Kur. forms below are in the full DEDR articles, not merely
# inferred from a displayed article head or another pending proposal.
add([1], 3149, "Kur. tākā 'air, wind, breeze' matches tākkā.")
add([4], 730, "Kur. ūxā 'darkness' matches the source's ūk?ā with its uncertain consonant sign.")
add([5], 1993, "Kur. xēxᵉl 'ground, earth, soil' matches kēkel 'earth'.")
add([7], 1514, "Kur. cicc 'fire' matches the survey's cʰīcʰ.")
add([13], 4567, "Kur. partā 'mountain, hill' matches pārtār 'mountain'.")
add([20], 2435, "Kur. cācā 'stone, pebble, rock' matches cācʰā.")
add([37], 399, "Kur. ālas 'adult male person' matches ālūs 'male'.")
add([46], 4841, "Kur. meccā 'high, tall, lofty' matches mecʰā 'tall'.")
add([58], 4575, "Kur. pōk 'ant' matches pook; this is the non-white-ant sense in the article.")
add([59], 5490, "Kur. berxā 'cat' matches berhā.")
add([71], 2135, "Kur. xolā 'tail' matches kʰolā.")
add([73], 773, "Kur. ekkā 'tortoise' matches ekā.")
add([75], 3589, "Kur. nalᵃkh 'work, action, labour' matches nālākʰānā 'work' with local nominal/verbal material.")
add([77], 1931, "Kur. xē̃s 'blood' matches hesso.")
add([83], 1977, "Kur. xebdā 'ear' matches hebdā.")
add([84], 1159, "Kur. xann 'eye' matches kʰān.")
add([86], 3585, "Kur. naṛī 'fever' matches nāṛī.")
add([89], 1943, "Kur. xeḍḍ 'foot, leg' matches hed.")
add([92], 1943, "Kur. xeḍḍ 'foot, leg' matches heād 'leg'.")
add([95], 1996, "Kur. xēsᵉr 'neck, shoulders' matches heser.")
add([97], 3064, "Kur. tatxā 'tongue' matches tātʰā.")
add([98], 3986, "Kur. pall 'tooth' matches pol.")
add([102], 5131, "Kur. mōjxā/mōsgā 'smoke' matches mojʰā.")
add([126], 2943, "Kur. ṭaṭxā 'mango' matches tāṭk?ā despite the source's uncertain consonant transcription.")
add([127], 688, "Kur. uynā 'to plough' matches ūynā.")
add([129], 4077, "Kur. pādā 'root' matches pādʰā.")
add([130], 530, "Kur. embā 'pleasant to the taste' and Malt. embe 'sweet' match embā 'sweet'.")
add([132], 4711, "Kur. mann 'tree' matches mānā.")
add([142], 4570, "Kur. pannā 'iron' matches pānnā.")
add([143], 5020, "Kur. munddh/mund 'first, ahead of' matches mūnḍā 'first'.")
add([147], 3655, "Kur. nāx 'four things' matches nāxh 'four'.")
add([155], 520, "Kur. esnā 'to break, divide by force' matches esnā.")
add([156], 2063, "Kur. xoṭṭnā 'to break, smash, pierce' matches kʰottūnā; the article also notes possible Indo-Aryan comparisons.")
add([157], 1100, "Kur. xaccnā 'to break by pulling, pull to pieces' matches kʰācʰnā 'break'.")
add([169, 270], 2952, "Kur. ṭāṭnā 'to lick, kiss' and ṭaṭṭ-ērnā 'to taste' match both source lick and taste responses.")
add([180], 5459, "Kur. bīsnā 'to sell' matches bēsnā.")
add([173], 3259, "Kur. tisⁱgnā 'to open a door or shutter' matches tīsīgnā.")
add([178, 252], 5549, "Kur. uinā 'to put down; retain, keep back' matches ūīnā in both put and keep contexts.")
add([203], 4065, "Kur. pāṛnā 'to sing' matches pārnā.")
add([204, 206], 231, "Kur. argnā 'to climb, mount, rise' matches both ascend and climb responses.")
add([205], 976, "Kur. ondᵒrnā 'to bring, take along' matches ondornā.")
add([186], 3323, "Kur. tuppnā 'to spit' matches tūpnā.")
add([198], 3991, "Kur. palknā 'to cut lengthwise, split' matches pālākʰnā 'split'.")
add([217], 1419, "Kur. kānā 'to go, progress' matches kānā 'move'.")
add([221], 3675, "Kur. ijnā 'to stand upright, rise to one's feet' matches ījñā.")
add([222], 1031, "Kur. ōgnā 'to swim, float' matches õgūnā.")
add([226], 5093, "Kur. mennā 'to eat grass, graze' matches meññā.")
add([227], 5093, "Kur. menta'ānā 'to graze, supply with grass' matches meñtānā.")
add([230], 2119, "Kur. xoynā 'to cut down grass with a sickle, reap' matches kʰoynā.")
add([231], 903, "Kur. ērnā 'to see' is the lexical stem in ērkān 'I saw'; first-person morphology remains within the source response.")
add([241], 145, "Kur. attnā 'to clothe or dress oneself' matches āttnā 'wear'.")
add([243, 264], 5516, "Kur. mennā 'to hear, listen, question, ask' matches the ask and hear responses.")
add([245], 786, "Kur. eṛnā 'to call, summon' matches eṛnā.")
add([244], 5087, "Kur. mēxnā 'to call, hail loudly' matches mēkʰnā.")
add([247], 2968, "Kur. ṭūṛnā 'to mark with dots or streaks' matches tūṛnā 'write' by the article's mark-making sense.")
add([251], 3714, "Kur. nuṛnā 'to hide, conceal' matches nūṛnā.")
add([259], 17, "Kur. axnā 'to know, realize' matches ākʰnā.")
add([263], 858, "Kur. elcnā 'to fear' matches elecʰnā.")
add([268], 1407, "Kur. xandrnā 'to sleep' matches kʰāndāvnā.")
add([272], 1225, "Kur. kappnā 'to feel with the hand, touch' matches kāppnā.")
add([278], 2001, "Kur. xēndnā 'to buy, purchase' matches kʰendnā.")
add([320], 5020, "Kur. muńjā 'extremity, beginning, headpoint, end' matches mū̃jjā 'end'.")
add([335], 5020, "Kur. munddh/mund 'before' matches mūndʰ.")
add([349], 3656, "Kur. nelā 'tomorrow' matches nelā.")
add([350], 4205, "Kur. pisā 'afterwards' matches pīsā.")
add([353], 1619, "Kur. kiyyā 'beneath, under' matches kīyyā 'down'.")
add([361, 362], 5151, "Kur. ekāsē 'of what kind' and related interrogatives match the source's two 'how' forms.")
add([356], 1, "Kur. ās 'that man' matches ās 'he'.")
add([357], 1, "Kur. ād 'that woman or thing' matches the feminine ādīge with local case material.")
add([358], 410, "Kur. isan/hisan 'here' matches īsān.")
add([363], 5160, "Kur. ēn 'I' matches ēn.")
add([364], 410, "Kur. īd 'this woman or thing' matches the source transcription ī : d 'it/this'.")
add([368], 5154, "Kur. ēm (oblique em-) 'we (exclusive)' is the pronoun base of emhāī 'our'.")
add([369], 4743, "Kur. mal/malā 'not' matches māllā.")
add([370], 410, "Kur. ennē 'such as this' matches enne 'so/thus'.")
add([371, 377], 1, "Kur. asan/hasan 'there, thither' matches āssān, including the source's distal that/there elicitation contexts.")
add([374], 1, "Kur. anti 'then, after this' matches āntīle with local suffixing.")
add([375, 376, 378], 1, "DEDR 1 gives Kur. ār/abṛar and abṛā distal plural persons/things; these are the source's those/they forms.")
add([381], 3647, "Kur. nām (oblique nam-/naŋg-) 'we (inclusive)' matches nām 'to us' with a case use.")
add([384], 5151, "Kur. endrā 'what thing' matches enḍṛā.")
add([386], 5151, "Kur. eksan 'in what place' matches exhān 'where'.")

hold([148, 149, 150, 151, 152], "Five through nine resemble nearby Indo-Aryan numeral forms in this Census transcription; immediate donors and indigenous alternatives need review.")
hold([235, 236], "The same pārmnā is glossed both 'bark' and 'bite (dog)'; DEDR 3962 has Kurux parᵃmnā 'bite' in a grinding-seed sense, but these animal-response meanings need source adjudication.")
hold([145], "Mūndā is also glossed 'first' at source index 143 and corresponds to Kurux mund 'first' in DEDR 5020; the separate 'third' gloss is likely a survey error and is held.")
hold([125], "Pātā 'leaf' resembles Indo-Aryan pattā; it does not match Kurux atxā in DEDR 141 closely enough to infer an inherited reflex or immediate donor.")
hold([347], "Pāīrī-bīrī 'morning' contains Kurux pairī in DEDR 3805, but the second constituent requires a separate verified parent before an ordered-component proposal.")
hold([76], "Mēl 'back' shares the gloss of Kurux mēd in DEDR 5099, but the final lateral versus stop correspondence is unverified in this survey transcription.")
hold([99], "Kātāmā 'waist' is only a partial resemblance to Kurux kaṛmā in DEDR 1143; the medial sequence needs confirmation.")
hold([305, 306], "The same ūjgo form is glossed 'soft' and 'straight'; Kurux ujgō 'straight' occurs in supplementary DEDR 583A, which has no current CLDF etymon node, and the conflicting source gloss needs review.")

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
            assert row["Entry_ID"] not in articles
            articles[row["Entry_ID"]] = {k: row[k] for k in ("Page", "Citation", "Raw_Article")}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, parents - set(articles)
(P / "proposed-decisions.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted,
                                                            held=research_holds), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(research_holds, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(dict(proposed=len(accepted), rows=len(accepted), held=len(research_holds), parents=len(parents)))
