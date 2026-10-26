"""Conservative, article-backed proposals for residual Maliyad Tamil Nadu Census records."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 450
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

# Full DEDR articles support the proposals below; source records remain unchanged.

add([1], 1499, "Kannada/Tulu gāli 'wind, air' matches gāli 'air'.")
add([8], 2929, "Tamil neruppu and Malayalam nerippu 'fire' match nerɨppɨ.")
add([10], 1438, "Kannada kāḍu and Tamil kāṭu 'forest' match kādu.")
add([14], 5503, "Tamil veḷḷam and Kannada beḷḷa 'flood' match vellã.")
add([15], 4742, "Tamil malai and Malayalam mala 'mountain' match mələ.")
add([18], 5381, "Tamil māṉam and Malayalam mānam 'sky' match mānəɔ̃.")
add([20], 1298, "Tamil kal and Malayalam kallu 'stone' match kiḷḷɨ within the stone family.")
add([23], 3045, "Tamil/Malayalam taṇṇīr/taṇṇi 'water' match təṇṇṇi; this is the cold-water compound in the full article.")
add([31], 131, "Tamil aṇṇaṉ and Kannada aṇṇa 'elder brother' match əṇṇəñ, despite the Census column's combined elder/younger gloss.")
add([32], 3085, "Tamil tampi and Malayalam tampi 'younger brother' match te̪mbi, despite the combined gloss.")
add([33], 4616, "Kannada magaḷ and Tamil makaḷ 'daughter' match məxə.")
add([51], 23, "Tamil akkā and Kannada akka 'elder sister' match əkkā, despite the combined elder/younger gloss.")
add([56], 4395, "Tamil peṇ/peṇṭu and western poṇḍ- 'wife, woman' forms match moṇḍɨ.")
add([59], 4355, "Tamil pūṉai and Kodagu pu·ñe 'cat' match pūnā in the article's broad cat family.")
add([61], 2818, "Tamil/Malayalam cēval 'cock' matches śævəḷ.")
add([63], 2901, "Tamil ñaṇṭu and Kodagu ñaṇḍï 'crab' match nənḍuŋa.")
add([64], 3650, "Tamil nāy and Kannada nāyi 'dog' match nāɉi.")
add([65], 4939, "Tamil muṭṭai and Malayalam muṭṭa 'egg' match muṭṭiŋgə.")
add([66, 80], 2591, "Kannada ṟekke and Kodagu rekke 'wing' match rəkkə in both feather and wing elicitation contexts.")
add([70], 4642, "Tamil mayil and Kannada maylu 'peacock' match mεyil.")
add([73], 1711, "Tamil kutirai and Kannada kudire 'horse' match kuðɨrā.")
add([76], 1769, "Kannada kōti and Telugu kōnti 'monkey' match gōntti within the whole article.")
add([78], 4307, "Tamil puli 'tiger' matches puli.")
add([79], 5155, "Tamil āmai and Kannada āme 'tortoise' match āmā.")
add([83], 3736, "Tamil neñcu 'breast, chest' matches neñcɨ.")
add([85], 5031, "Tamil mūñci and Malayalam muññi 'face' match mūñɈī.")
add([87], 5409, "Tamil viral and Malayalam viral 'finger' match verakkā in this finger family.")
add([95], 5352, "Tamil vāy and Malayalam vāy 'mouth' match vāɉī.")
add([98], 3986, "Tamil pal and Malayalam pallu 'tooth' match pilli.")
add([101], 2674, "Tamil uppu and Malayalam uppu 'salt' match uppvɨ.")
add([102], 4240, "Tamil pukai and Malayalam puka 'smoke' match puxā.")
add([110], 3140, "Tamil taṟi and Malayalam taṟi 'weaver's loom' match təri.")
add([111, 336], 2237, "Tamil kōl and Kannada kōlu 'stick' match both kōl/kōlu source records.")
add([113], 5393, "Tamil vīṭu 'house, habitation' matches vvḍɨ within the full leave/house article.")
add([115], 2207, "Kannada gōḍe 'wall' matches gōḍā.")
add([117], 5028, "Tamil mūṅkil and Tulu muṅgilů 'bamboo' match muŋile; the immediate lect is not specified in the Census.")
add([124], 3549, "Tamil tōṭṭam and Kannada tōṭa 'garden' match tōṭṭəm.")
add([129], 3753, "Tamil nel and Kannada nellu 'paddy' match nɛllu.")
add([141], 5288, "Tamil valai and Malayalam vala 'net' match vale.")
add([147, 148], 1787, "Tamil kuruṭu and Kannada kuruḍa/kuruḍi 'blind' match the masculine/feminine kuruṭa/kuruṭi responses.")
add([153], 3821, "Tamil paccai 'green' matches paccakitu with local adjective morphology.")
add([156], 4322, "Tamil puḷippu 'sourness' matches puḷippu.")
add([157], 5496, "Tamil veḷḷai 'white' matches veliλeṇṭu with local adjective morphology.")
add([158], 4635, "Tamil mañcaḷ 'turmeric, yellow' matches məñcɨlu.")
add([182], 486, "Tamil irumpu and Malayalam irumpu 'iron' match irumbɨ.")
add([185], 5496, "Tamil veḷḷi 'silver' matches vɛḷḷi within the whole white/silver article.")
add([186], 4950, "Tamil mutal 'beginning, first' matches muðəlu.")
add([189], 2826, "Tamil añcu 'five' matches añṭśɨ.")
add([190], 910, "Tamil/Malayalam ēḻu 'seven' matches ēgɨ in the regional numeral family.")
add([191], 784, "Tamil eṭṭu and Malayalam eṭṭu 'eight' match eṭḍɨ.")
add([193], 3918, "Tamil pattu 'ten' matches pᴐtðɨ.")
add([197], 946, "Tamil oṭi/uṭai 'break' matches oṭikertu with the Maliyad verbal ending.")
add([206], 77, "Tamil aṭi 'beat, strike' matches aṭioθœ.")
add([209], 3570, "Tamil/Malayalam nakku 'lick' matches nakkerððu.")
add([219], 5421, "Tamil vil and Malayalam vilkka 'sell' match vikirððu within the full sale article.")
add([224], 3323, "Tamil/Malayalam tuppu 'spit' matches tupparθu.")
add([226, 231, 335], 1624, "Tamil kīṟu 'tear, cut, gash' matches the kī-/ki- cutting forms in these elicitation contexts.")
add([227], 859, "Tamil eṟi 'throw' matches erikerððu.")
add([232], 3549, "Tamil tōṇṭu 'dig' matches toṇṛaððu.")
add([242], 4065, "Tamil pāṭu 'sing' matches paððuŋgo.")
add([252], 4020, "Tamil paṟa 'fly' matches parakutu.")
add([255], 2053, "Tamil koṭu and Kannada koḍu 'give' match kuṭukkutu.")
add([256], 4572, "Tamil pō 'go' matches porððu within the whole article.")
add([260], 1041, "Tamil ōṭu 'run' matches orððu.")
add([264], 3687, "Tamil nīntu 'swim' matches nienθarððu.")
add([268], 3582, "Tamil naṭa 'walk' matches naṭakerððu.")
add([280], 1124, "Tamil kaṭi 'bite' matches kaṭikertu.")
add([282], 1654, "Tamil kuṭi 'drink' matches kuṭikerððu.")
add([283], 3263, "Tamil tiṉ 'eat' matches tiŋkerððu.")
add([284], 600, "Tamil ūṭṭu and Malayalam ūṭṭuka 'feed' match uṭṛaððu.")
add([291], 2855, "Tamil col 'say' matches ∫ollarððu.")
add([302], 1728, "Tamil kuntu 'sit, squat' matches konθukerððu.")
add([309], 3419, "Tamil teri 'know' matches terinjaθu.")
add([315], 2017, "Tamil kēḷ 'hear' matches kekkuθu within the full article.")
add([334], 1443, "Tamil kāṭṭu 'show', a causative of kāṇ 'see', matches kaṇbikkerððu in the full see/show article.")
add([342], 4411, "Tamil periya 'big' matches pεrāðɨ with local adjective morphology.")
add([345], 4992, "Tamil muḻu 'whole, full' matches mulsə̄ in the full/complete article.")
add([357], 2594, "Tamil ciṉṉa 'small' matches cinnnðɨ with local adjective morphology.")
add([360], 3772, "Tamil nēr 'straight' matches nērāxə.")
add([384], 5020, "Tamil muṉṉē and Malayalam munnē 'before' match minnē.")
add([393], 2552, "Tamil rā and Malayalam rā 'night' match rāvārɨ within the full night article.")
add([395], 3758, "Tamil nerunai and Malayalam innalē 'yesterday' match nēsɨrɨ in this comparative family.")
add([411], 5160, "Tamil nāṉ and Kannada nān 'I' match nānɨ.")
add([419], 2559, "Tamil illai 'not' matches illā.")
add([421, 425], 1, "Tamil a-/atu 'that' and distal aṅku 'there' match aðɨŋgə and aŋgi within the full demonstrative article.")
add([431, 432, 433, 434, 435], 5151, "The full interrogative article includes Tamil entu 'what', eppōtu 'when', eṅkē 'where', and yār 'who/whom', matching these Maliyad questions.")
add([442], 698, "Tamil uḷ and Malayalam uḷḷu 'inside' match ulle.")
add([443], 240, "Tamil alai and Malayalam ala 'wave' match aleŋkā within the wave/motion article.")
add([450], 1182, "Tamil kaṇṇāṭi and Kannada kannaḍi 'mirror' match kəṇṇāḍi.")

hold([34], "āvā ‘father’ resembles the avva forms for 'mother' in DEDR 273; kinship gloss conflicts with that article.")
hold([45], "ariyāgālu ‘twin’ recurs as the response for 'man'; source meaning needs review before family assignment.")
hold([57], "pɔnnɨtāɉi ‘woman’ has a clear woman-root opening but the second element is unresolved; hold the whole response.")
hold([62], "pāsumāḍɨ ‘cow’ appears compound or contact-shaped; component history and immediate donor are unresolved.")
hold([69], "pōtt̪ā̪du ‘goat’ appears compound; whole-form segmentation and component ancestry are unresolved.")
hold([75], "penmuṭa ‘louse’ has a second element that the proposed louse article does not explain.")
hold([155], "erra ‘red’ resembles Telugu and Central Dravidian DEDR 865, but immediate donor into this Tamil Nadu variety is unresolved.")
hold([181], "məñṭśulɨ ‘gold’ has an unclear second element; a colour or metal-root match alone is insufficient.")
hold([194], "itti ‘twenty’ is too short and divergent for a secure assignment from numeral-family comparisons.")
hold([198, 294], "The same source response neruppil nikkuθu is glossed both 'burn' and 'borrow'; source sense is inconsistent.")
hold([244, 246], "malaλerθu/malaerθu ‘ascend/climb’ appears compound ('mountain' + rising verb); no secure whole-form analysis yet.")
hold([267, 337], "The same response vānti is glossed both 'visit' and 'vomit'; source sense and donor path need review.")
hold([344], "mɛttɛ ‘full’ differs from the secure mulsə̄ response and resembles a bedding term; source sense requires review.")
hold([377], "nεḍɨvɨli ‘middle’ looks compound; the second element and whole-form analysis need review.")
hold([403, 408], "keru is elicited for personal and possessive pronouns but may denote a person/man; gloss and morphology unclear.")
hold([404, 409], "poṇṇu is elicited for personal and possessive pronouns but is also a woman noun; gloss and morphology unclear.")
hold([418, 430], "nəmbɨrɨ is glossed with inclusive/exclusive 'we/our' ambiguity; pronominal family assignment requires paradigm review.")
hold([426], "aθuŋ aṭṭeru ‘they’ appears compound; morphology and whole-form ancestry need review.")

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
