"""Conservative, article-backed proposals for residual Kannada Census responses."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 438
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

# Comparanda below are from full DEDR articles. Each proposed edge remains
# pending and does not edit accepted CLDF etymology assignments.
add([7], 1514, "Kannada kiccu 'fire' matches kicū.")
add([9], 1438, "Kannada kāḍu 'forest, jungle' matches kāṭu.")
add([14], 4742, "Kannada male 'mountain, forest' matches ɱəlē 'mountain'.")
add([17], 5033, "Kannada mōḍa 'cloud, cloudy day' and the article's Tamil 'clouded sky' and Kui 'cloud, sky' senses support ɱɔ̄ḍā in the source's broad 'sky' elicitation.")
add([31], 3085, "Kannada tamma 'younger brother' matches təɱma; the Census combines elder/younger in its gloss, but this response has the younger-brother stem.")
add([35], 1173, "Kannada gaṇḍu 'male, man' matches kɛṇḍu.")
add([53], 4616, "Kannada maga 'son' matches məŋgā.")
add([55], 4395, "Kannada peṇḍati and related female/wife forms match keṇḍru 'wife'.")
add([56], 4395, "Kannada peṇṇu/heṇṇu 'woman, female' matches kɛnnākā with local suffixing.")
add([61, 67], 2248, "Kannada kōr̤i 'cock, hen, fowl' matches both kɔ̄ḷi responses.")
add([64], 2901, "The article gives Tamil ñaṇṭu and Kodagu ñaṇḍï 'crab'; nanḍo is the corresponding crab term.")
add([65], 3650, "Kannada nāyi 'dog' matches nāyē.")
add([66, 80], 2591, "Kannada eṟaṅke 'wing' and article-wide wing/feather forms match erəŋgəlu and yeraŋgalu.")
add([69], 3606, "Kannada nari 'jackal, fox' matches nɛri 'fox'.")
add([72], 4642, "Kannada maylu 'peacock' matches ɱəy.lu.")
add([74], 1711, "Kannada kudire/kudare 'horse' matches kutərə.")
add([79], 5155, "Kannada āme 'tortoise' matches āɱe.")
add([96], 1366, "Kannada kattu 'neck, throat' and Tamil kaḻuttu 'neck' belong to the full article containing kəluttu.")
add([104], 215, "Kannada akki 'husked rice' matches əkkī.")
add([121], 3408, "Kannada teṅgu 'coconut palm' and Kodagu teŋge 'coconut' match teŋgɛ̄.")
add([124], 3549, "Kannada tōṭa/tōṇṭa 'garden' matches tōṭṭā.")
add([133], 5535, "Kannada bēr(u) 'root' matches pɛ̄ru.")
add([147], 1787, "Kannada kuruḍa 'blind man' matches kuruḍā.")
add([152], 2654, "Kannada sūḍu/sūṭe 'burning, hot' matches sū.ṭāyṭ.ṭu with local adjectival morphology.")
add([158], 220, "Kannada arisina/arsiṇa 'turmeric' and Toda arsn 'yellow' match arsəṇā as a turmeric-based yellow term in the whole article.")
add([184], 5496, "The article gives Kannada beḷḷi 'silver'; pɛllī preserves that silver noun.")
add([188], 474, "Kannada eraḍu 'two' matches yɛruḍu.")
add([189], 3655, "Kannada nāku 'four' matches nakku.")
add([190], 2826, "Kannada aydu 'five' matches aytu.")
add([206], 3570, "Kannada nakku 'to lick' matches nakkutu with a local infinitive ending.")
add([216], 4834, "Kannada māṟu 'to barter, sell' matches mārutu.")
add([221], 3323, "Kannada tūpu 'to spit' matches tūpputu.")
add([223, 228], 1624, "The article includes Kodagu ki·r- 'to tear' and Tulu kīruni 'to split'; the same kīrutu was elicited for tear and cut.")
add([229], 3549, "Kannada tōḍu 'to dig' matches tɔ̄nḍutu.")
add([235], 2132, "Kannada kollu 'to kill' matches kɔllutu.")
add([245], 347, "Kannada āḍu 'to dance' matches āḍḍəkārā with a local verbal formation.")
add([256], 1041, "Kannada ōḍu 'to run' matches ōḍutu.")
add([264], 3582, "Kannada naḍe 'to walk' matches neḍivuətu.")
add([274], 5401, "Kannada bittu 'to sow seed' matches bitutu.")
add([279], 1147, "Kannada kaṭṭu 'to bind, tie' matches kɛḍḍutu.")
add([289], 2426, "Kannada sāy 'to die' matches sāyvətu.")
add([297], 4834, "Kannada māṟu 'to change, exchange' matches ɱāḍrutu 'change' with the source's rhotic/stop transcription uncertainty.")
add([302], 3419, "Tamil teri and Kannada tiḷi within the full know/perceive article match terivətu 'to know'.")
add([308], 2017, "Kannada kēḷu 'to hear, listen' matches kɛ̄lutu.")
add([311], 707, "Kannada oṟaṅgu 'to lie down, rest, sleep' matches ɔrəŋgutu.")
add([331], 990, "DEDR 990 includes Kannada oneness/unity and western okka 'all together' forms; ɔkkā is the all/whole response.")
add([335], 1148, "Kannada gaṭṭi 'firmness, hardness' matches kiṭṭəā 'hard'.")
add([325], 1443, "Kannada kāṭṭu 'to show' occurs in this seeing/showing article and matches kāṭṭutu.")
add([326], 1624, "The same kīrutu elicited for tear/cut is used for scratch; DEDR 1624 includes tear, gash and slit senses.")
add([328], 1477, "Kannada kāṟ 'to vomit' matches kārutu.")
add([361], 8, "Kannada agala 'width, spaciousness' matches agla 'wide'.")
add([374], 5020, "Kannada mun/munnu 'before, in front' matches ɱuntə.")
add([376], 3805, "Kannada pagal 'daytime, a day' and nearby Badaga bahalu match kəhalu 'day'.")
add([385], 3758, "Kannada ninne 'yesterday' matches mɛṇṇɛ̄ via the western ñenne/ninne form.")
add([393], 1, "Kannada ava 'that man/woman' matches avā 'he/she'.")
add([395], 410, "Kannada illi 'here' matches illī.")
add([396], 1, "Kannada distal ava forms provide the base of auvā 'his/her'.")
add([398], 5160, "Kannada nān 'I' matches nā.")
add([404], 5160, "Kannada nan-/en- oblique first-person base matches nənttu 'my'.")
add([405], 5154, "Kannada ām (oblique em-) and Kodagu naŋga 'we' give the first-person plural base of nəŋgā 'our'.")
add([406], 2559, "Kannada illa 'not, absent' matches illɛ̄.")
add([411], 1, "Kannada alli 'there' matches allī within the distal demonstrative article.")
add([412, 413], 1, "The full distal plural article includes western avar/ākuḷu forms; ak.ku.lum 'they' and akkulutu 'their' preserve that plural base.")
add([417], 5151, "Kannada yā/ē interrogative base 'what' matches yānā.")
add([419], 5151, "Kannada elli 'where' matches yɛlli.")
add([420], 5151, "Kannada yār 'who' matches yārā.")
add([423], 3688, "Kannada nīngaḷ 'you' and oblique nim- underlie niŋgulutu 'your'.")
add([438], 1182, "Kannada kannaḍi 'mirror' matches kəṇṇāḍi exactly within this whole article.")

# The Census data contain visibly inconsistent elicitation/gloss alignments.
# Keep conflicts and incomplete compounds explicit rather than making a link
# from a familiar-looking fragment alone.
hold([2], "Pūdi 'ashes' resembles Kannada būdi in DEDR 4316, whose bracketed forms are probably Sanskrit bhūti loans; immediate donor unverified.")
hold([59], "Pūjē 'cat' matches non-Kannada cat reflexes in DEDR 4355 but may be a regional borrowing; the immediate lect is unverified.")
hold([76], "Keṇu 'louse' has the right gloss for DEDR 4449 but differs from Kannada pēn and neighboring forms in its initial consonant; correspondence needs verification.")
hold([78], "Nɛri is independently elicited as 'fox' at index 69 and matches Kannada nari 'fox' in DEDR 3606; the separate 'tiger' gloss conflicts and needs source review.")
hold([102], "Gālu is glossed 'milk' despite resembling western kālu 'leg'; a milk etymology from this spelling is unsafe.")
hold([98], "Gəllu 'tooth' differs in its initial consonant from Kannada pal and other DEDR 3986 tooth forms; the source transcription needs verification.")
hold([116], "Muŋgil 'bamboo' occurs in DEDR 5028 for Tamil/Tulu/Irula but not Kannada; the immediate donor into this survey lect is unverified.")
hold([134], "Muṭṭai is glossed 'sweet' despite ordinary regional egg sense; source gloss needs verification.")
hold([136], "Sakkara is glossed 'wheel' despite a common sugar form; source gloss/contact source unresolved.")
hold([148], "Nīlā 'blue' resembles Indo-Aryan nīla-type vocabulary; immediate donor into this lect is unverified.")
hold([191], "Gəttu 'ten' differs sharply from Kannada pattu and other DEDR 3918 forms; numeral transcription needs verification.")
hold([181], "Irumbu 'iron' is present in DEDR 486 for Tamil/Malayalam/Kodagu but not Kannada; the source response may be borrowed, with immediate lect unknown.")
hold([239], "Bāḍḍupāḍutu 'sing' has an extra leading constituent before the DEDR 4065 pāḍu stem; complete ordered components remain unresolved.")
hold([253, 260], "The go/swim gloss is plausible, but the source stem differs from checked Kannada and neighboring DEDR forms; the exact sound correspondence needs verification.")
hold([309], "Jiripputu 'laugh, smile' has no demonstrated phonetic relationship to Kannada nagu in DEDR 3569; no secure parent assigned.")
hold([314], "Muṭṭutu 'touch' has no demonstrated whole-stem relationship to Kannada toḍu in DEDR 3480; no secure parent assigned.")
hold([339, 387], "The compared new/behind family shares a broad sound pattern, but the source initial consonant differs from checked DEDR reflexes; retain for source transcription review.")
hold([399], "Adʰu is a distal demonstrative under DEDR 1, while the source gloss says 'it/this'; proximal versus distal deixis requires source adjudication.")
hold([416], "Nəŋgā 'we' could be inclusive DEDR 3647 or exclusive DEDR 5154; the source does not specify clusivity.")

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
