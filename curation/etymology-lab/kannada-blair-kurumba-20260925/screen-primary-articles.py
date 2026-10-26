"""Conservative article-by-article screen of exploratory form/gloss concordances."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
candidates = json.loads((P / "cross-pass-candidates.json").read_text())
articles = json.loads((P / "candidate-article-evidence.json").read_text())

# Each note was checked against the complete cached DEDR article, not its display head.
comparanda = {
    "d4797": "Kannada māḍu ‘do, make, perform’",
    "d4307": "Kannada puli ‘tiger’",
    "d2674": "Kannada uppu ‘salt’",
    "d3085": "Kannada tamma ‘younger brother’",
    "d1": "the Kannada a- remote demonstrative and avar ‘they’",
    "d2591": "Kannada ṟekke ‘wing’",
    "d3650": "Kannada nāy/nāyi ‘dog’",
    "d131": "Kannada aṇṇa ‘elder brother’",
    "d23": "Kannada akka ‘elder sister’",
    "d410": "the Kannada i- proximate demonstrative",
    "d5151": "the Kannada interrogative base e-/yā- and its where/what forms",
    "d99": "Kannada aṭṭe ‘leech’",
    "d3302": "Kannada toḍe ‘thigh’",
    "d178": "Kannada ambu ‘arrow’",
    "d5024": "Kannada mūgu ‘nose’",
    "d2901": "Tamil naṇṭu and Kannada naḷḷi ‘crab’",
    "d2826": "Kannada ay(i)du ‘five’",
    "d3876": "Kannada paṭṭe ‘tree bark’",
    "d5352": "Kannada bāy(i) ‘mouth’",
    "d3647": "Kannada nāvu ‘we (inclusive)’",
    "d2207": "Kannada gōḍe ‘wall’ in article subsection (b)",
    "d1366": "Kannada kattu ‘neck’",
    "d3582": "Kannada naḍe ‘walk’",
    "d3564": "Tamil tōḷ ‘shoulder’ and Kannada tōḷ(u) ‘arm’",
    "d2115": "Kannada kombu ‘animal horn’",
    "d3759": "Kannada netti ‘forehead’",
    "d2023": "Kannada kayyi ‘hand/arm’",
    "d1438": "Palu Kurumba kāḍu ‘field or garden in the forest’",
    "d5288": "Kannada bale ‘net’",
    "d1355": "Kannada gadde ‘paddyfield’",
    "d3549": "Kannada tōḍu ‘dig/excavate’",
    "d5422": "Kannada billu ‘bow’",
    "d3015": "Kannada taṅge ‘younger sister’",
    "d833": "Kannada eli ‘rat/mouse’",
    "d990": "Tamil oru and Kannada obba ‘one person’",
    "d946": "Tamil ōṭṭai ‘hole’ and Kannada ōṭe ‘crack’",
    "d3655": "Kannada nāku ‘four’",
    "d916": "Kannada ēṟu ‘rise/ascend’",
    "d1147": "Kannada kaṭṭu ‘bind, tie’",
    "d1718": "Kannada kuttige ‘neck/throat’",
    "d393": "Tamil āvi ‘soul, breath’",
    "d5161": "Kannada āne ‘elephant’",
    "d5087": "Kannada mēke ‘she-goat’",
    "d1682": "Kannada guḍḍa ‘hill’",
    "d1977": "Kannada kivi ‘ear’",
    "d5401": "Kannada bittu ‘sow seed’",
    "d5073": "Kannada may(i) ‘body’",
    "d2495": "Kannada cikka ‘small’",
}
cross_article_ids = {x["parent"] for x in candidates}
assert set(comparanda) <= cross_article_ids <= set(articles)

# These exact strings would overstate an unresolved borrowing, sound path, or
# multi-component compound if linked from a single article in this checkpoint.
exclusions = {
    "d4096": "milk: initial p-/h-/zero pathway and immediate donor need review",
    "d4085": "snake: initial p-/h-/zero pathway and immediate donor need review",
    "d4449": "head louse: yenu versus article pēn needs a sound-path check",
    "d3918": "eleven: a transparent ten-plus-one compound needs component analysis",
    "d4711": "mārapatte bark: tree-plus-bark compound needs component analysis",
    "d3805": "daytime: source agalu versus article pagal needs initial-consonant review",
    "d2410": "son-in-law: āṛṛyā versus aḷiya needs sound-path review",
    "d282": "cry: āḷu versus article ar̤u needs sound-path review",
    "d5154": "exclusive we: nā̃a versus article ā(m) needs morphology review",
    "d546": "liver: yiri versus article īral needs sound-path review",
    "d4410": "name: pēru versus Kannada pesar(u) needs source/donor review",
    "d5370": "bat: vāvvāḷ versus Kannada bāvali needs source/donor review",
    "d5031": "face: mūndzī has only cross-dialectal comparanda here; donor needs review",
    "d4088": "mat: pāyɨ has only cross-dialectal comparanda here; donor needs review",
    "d2054": "axe: koḍāli versus Telugu sickle requires sense and donor review",
}
assert set(comparanda).isdisjoint(exclusions)
assert set(comparanda) | set(exclusions) == cross_article_ids

# Additional direct homepage hits whose form and sense were checked in their
# complete DEDR articles. Index lists refer to the exact residual inventory.
homepage_additions = {
    "d1355": ("Kannada gadde ‘wet paddyfield’", [13, 616, 701, 758, 1187]),
    "d833": ("Kannada eli/ili ‘rat’", [57, 153, 640]),
    "d1224": ("Kannada kappe and dialect keppe ‘frog’", [64, 157, 428, 808, 1392]),
    "d990": ("Kannada ondu ‘one thing’", [170]),
    "d410": ("Kannada īga ‘now’ within the proximate demonstrative family", [177]),
    "d1": ("Kannada ā ‘that’ within the remote demonstrative family", [210]),
    "d386": ("Kannada āl ‘cry aloud’", [217]),
    "d3683": ("Kannada nenasu ‘think’", [238]),
    "d5538": ("Kannada bēli ‘fence’", [256, 332]),
    "d4616": ("Kannada magu ‘child’", [259, 345, 620, 671]),
    "d5073": ("Kannada may(i) ‘body’", [261]),
    "d3582": ("Kannada naḍi ‘walk’", [311, 489]),
    "d1366": ("Kannada gaṇṭalu ‘throat’", [351]),
    "d4157": ("Kannada heṇa ‘corpse’", [366]),
    "d1407": ("Kannada kanasu ‘dream’", [371]),
    "d4649": ("Kannada himmaḍi ‘heel’", [378]),
    "d5486": ("Kannada bevaru ‘sweat’", [383]),
    "d4626": ("Kannada maṅga ‘monkey’", [409]),
    "d5155": ("Kannada āme ‘turtle’", [423]),
    "d2902": ("Kannada navilu ‘peacock’", [458]),
    "d1584": ("Kannada giṇi ‘parrot’", [459]),
    "d1871": ("Kannada gūge ‘owl’", [460]),
    "d3486": ("Kannada toṭlu ‘cradle’", [466]),
    "d3309": ("Kannada duṇḍage ‘round’", [485]),
    "d3687": ("Kannada hīju ‘swim’", [490]),
    "d4620": ("Kannada makkari ‘basket’", [662]),
    "d1873": ("Kannada kūsu ‘infant/child’", [845, 930, 1024, 1117, 1190, 1264, 1352, 1426]),
    "d1911": ("Kannada kūr̤ ‘cooked rice, food’", [894, 1079, 1232]),
    "d1884": ("Kannada gūḍe/gūḍa ‘basket’", [911, 1098, 1247, 1409]),
    "d4715": ("Kannada mom-magaḷu ‘granddaughter’", [936, 1433]),
    "d1249": ("Kannada kayyi ‘bitter’", [1294]),
    "d2680": ("Kannada suyilu ‘breath’", [1371]),
    "d1114": ("Kannada kaḍave ‘elk/large deer’", [1388]),
    "d3268": ("Kannada siyyane ‘sweet’", [1456]),
}
assert set(homepage_additions) <= set(articles)
homepage = json.loads((P / "homepage-candidates.json").read_text())
assert len(homepage) == 1504

# The report itself supplies parallel p-, h-, and vowel-initial responses in
# its Kannada Kurumba lists. These witnesses, together with the complete DEDR
# articles, resolve several previously deferred initial-consonant candidates.
sound_series = {
    "d4096": ("milk", "Kannada pāl ‘milk’; this same survey also records hāḷu/hālu ‘milk’"),
    "d4085": ("snake", "Tamil pāmpu and Kannada pāvu ‘snake’; this same survey also records hāvu and pāmbu ‘snake’"),
    "d3986": ("tooth", "Kannada pal ‘tooth’; this same survey also records hallu/hāḷḷu ‘tooth’"),
    "d4345": ("flower", "Kannada pūvu/puvvu ‘flower’; this same survey also records huvu ‘flower’"),
    "d4307": ("tiger", "Kannada puli ‘tiger’; this same survey has pulī, hulī, vulī, and ulī ‘tiger’"),
    "d4039": ("pig", "Kannada pandi ‘pig’; this same survey also records haŋḍī ‘pig’"),
    "d3805": ("day(time)", "Kannada pagal ‘daytime’; this same survey also records hagulu ‘daytime’"),
    "d4494": ("stomach", "Kannada poṭṭe ‘stomach’; this same survey also records hɔtte ‘stomach’"),
    "d3918": ("ten", "Kannada pattu ‘ten’; this same survey also records pāttu and h-less attɨ ‘ten’"),
    "d4303": ("field, dry", "Kannada pola ‘plough-field’; this same survey has halā, valā, and olā for dry field"),
    "d5224": ("hit", "Kannada baḍi ‘hit’; this same survey has baḍi/bāḍi and aḍi for ‘hit’"),
    "d4449": ("louse, head", "Kannada pēn ‘head-louse’; this same survey has henu, enu, and yenu for head-louse"),
}
assert set(sound_series) <= set(articles)

selected = []
deferred = []
for row in candidates:
    index, parent = row["index"], row["parent"]
    if parent == "d4307" and row["form"] not in {"pulī"}:
        deferred.append({**row, "reason": "tiger: non-pulī forms need initial-consonant/donor review"})
    elif parent == "d2901" and row["form"] != "naḷḷi":
        deferred.append({**row, "reason": "crab: nāṇḍu matches Tamil/Malayalam rather than article Kannada; donor needs review"})
    elif parent == "d4797" and row["gloss"] == "cook":
        deferred.append({**row, "reason": "māḍu ‘cook’ is broader than the article’s do/make wording"})
    elif parent in comparanda:
        selected.append({**row, "comparandum": comparanda[parent]})
    else:
        deferred.append({**row, "reason": exclusions[parent]})

selected_indexes = {x["index"] for x in selected}
records = json.loads((P / "survey-inventory.json").read_text())
for parent, (note, indexes) in homepage_additions.items():
    for index in indexes:
        assert index not in selected_indexes, index
        assert any(c["parent"] == parent for c in homepage[index - 1]["candidates"]), (index, parent)
        record = records[index - 1]
        selected.append({"index": index, "recordId": record["ID"], "form": record["Form"],
                         "gloss": record["Gloss"], "parent": parent,
                         "comparandum": note, "route": "homepage-exact-form"})
        selected_indexes.add(index)

series_witnesses = {}
with (DATA / "cldf/forms.csv").open() as source:
    compiled = [row for row in csv.DictReader(source)
                if row["Language_ID"] == "Kannada" and
                row["Source"].startswith("blairetal2012kurumba[")]
assert len(compiled) == 2507
for parent, (gloss, note) in sound_series.items():
    witnesses = [{"recordId": record["ID"], "form": record["Form"],
                  "gloss": record["Gloss"], "source": record["Source"]}
                 for record in compiled if record["Gloss"] == gloss]
    assert witnesses
    series_witnesses[parent] = {"gloss": gloss, "comparandum": note,
                                "sourceWitnesses": witnesses}
    for index, record in enumerate(records, 1):
        if record["Gloss"] != gloss or index in selected_indexes:
            continue
        if parent == "d3805" and index == 669:
            continue  # aṭṭɨlɨ is not secure in the pagal/hagulu series.
        if parent == "d4449" and index == 54:
            continue  # śēnū needs an independent initial-consonant analysis.
        selected.append({"index": index, "recordId": record["ID"],
                         "form": record["Form"], "gloss": record["Gloss"],
                         "parent": parent, "comparandum": note,
                         "route": "internal-source-sound-series"})
        selected_indexes.add(index)
for index in (279, 791, 1055, 1215):
    record = records[index - 1]
    assert record["Gloss"] == "louse, body" and record["Form"] == "yenu"
    assert index not in selected_indexes
    selected.append({"index": index, "recordId": record["ID"], "form": record["Form"],
                     "gloss": record["Gloss"], "parent": "d4449",
                     "comparandum": sound_series["d4449"][1],
                     "route": "internal-source-sound-series"})
    selected_indexes.add(index)
selected.sort(key=lambda row: row["index"])
deferred = [row for row in deferred if row["index"] not in selected_indexes]
(P / "source-sound-series.json").write_text(json.dumps(series_witnesses, ensure_ascii=False, indent=2) + "\n")

(P / "primary-screen-candidates.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
(P / "primary-screen-deferred.json").write_text(json.dumps(deferred, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"selected": len(selected), "articles": len({x['parent'] for x in selected}), "deferred": len(deferred)}))
