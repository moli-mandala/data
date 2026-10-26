"""Article-backed pending links and explicit holds for residual Canvin Markodi forms."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 149
proposed = {}
holds = {}

def add(indices, number, comparison):
    for i in indices:
        assert i not in proposed
        proposed[i] = (number, comparison)

def hold(indices, reason):
    for i in indices:
        assert i not in holds
        holds[i] = reason

add([1], 4992, "Tamil muṟu and Malayalam muṟuvan 'all, whole' match Markodi muḻɔn 'all' in the full article.")
add([4, 5, 6], 864, "Malayalam uṟumpu and Kodagu urupï 'ant' match Markodi urumbu variants.")
add([10], 2054, "The full curved-implement article includes Telugu koḍali 'sickle', matching Markodi kodāli 'axe'.")
add([23, 41, 43], 1594, "Malayalam ceṟumi 'girl' directly matches Markodi cerumi/cɜrumi in the small/young article.")
add([66, 67, 68], 5160, "Tamil eṉ- and Tulu ēnu 'I' match Markodi enu in the first-person article.")
add([83, 84, 85], 156, "Tulu appè and Telugu appa 'mother' match Markodi āppe/appe/appa within the whole kinship article.")
add([100, 101, 102], 3422, "Tamil teru 'street, highway' and Belari teru 'way' match Markodi taru/teru 'path'.")
add([119, 120, 121], 3009, "Tamil/Kannada takkāḷi 'tomato' match Markodi takkāḷi/takkḷi/takkaḷi in the full article; the word may have spread areally.")
add([125, 127], 3647, "Tamil nām with oblique nam- and Kannada nāvu 'we' match Markodi nemma/namma 'we two'; clusivity is not specified in the source.")
add([137, 138, 140, 141, 142], 3688, "Tamil nīṅkaḷ, Malayalam niṅṅaḷ, and Kannada nīngaḷ 'you plural' match Markodi nikkar/nigaḷ variants.")
add([143, 144, 145], 3684, "Tulu ī and Tamil nī 'you singular' match Markodi iy/ije/iyy within the second-singular article.")

hold([2, 3, 129, 130, 131], "pātta/pātte/pɛṭā is glossed 'all' or 'whole', but the direct whole-form DEDR family is not established.")
hold([7, 8, 9], "The ash responses appear compound-like (veṇ-/beṇ- plus nīr), and require ordered component analysis.")
hold([11, 12, 13], "The 'bad' responses may combine a negation or 'not good' with eḍḍe; the entire construction is unresolved.")
hold([19, 20], "DEDR 2049 has Malayalam koṭṭu 'head of a bone', but Markodi koṭṭa is glossed 'bone' without that narrower sense.")
hold([24, 25, 26, 27, 28, 29, 45, 46, 47, 99], "Recent crop or vegetable name; immediate donor and adaptation are unresolved.")
hold([30], "kannukālɨ 'cow' appears to contain calf/leg components; the whole response needs analysis.")
hold([51, 52, 53, 54, 55, 56, 57, 58, 59], "The elicitation is a finite clause; its inflection and whole response need analysis before a root assignment.")
hold([90, 91, 92], "anḍɨ/andi 'night' is close to regional 'evening' forms, but a secure whole-article path is lacking.")
hold([103, 104, 105], "maśavillɨ 'rainbow' is an overt compound; ordered components require separate evidence.")
hold([128], "tiŋgaḷ is 'week' here whereas DEDR 3213 directly records moon/month; the calendar sense path needs review.")
hold([136], "kollam 'year' may be a regional calendar term; the immediate source and article are unresolved.")
hold([139], "nikkerellam contains an overt additional quantifier; the whole response needs compound analysis.")
hold([146, 147, 148, 149], "aniya- 'younger sibling' has no secure whole-article match in the checked DEDR cache; do not infer from phonetic similarity alone.")

for i, record in enumerate(records, 1):
    if i not in proposed and i not in holds:
        holds[i] = (f"{record['Form']} ‘{record['Gloss']}’: no secure whole-form DEDR article "
                    "identified from the checked primary evidence; retain for further research.")
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
            assert row["Entry_ID"] not in articles, row["Entry_ID"]
            articles[row["Entry_ID"]] = {k: row[k] for k in ("Page", "Citation", "Raw_Article")}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents, parents - set(articles)
(P / "proposed-decisions.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted,
                                                            held=research_holds), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(research_holds, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
print(dict(proposed=len(accepted), rows=len(accepted), held=len(research_holds), parents=len(parents)))
