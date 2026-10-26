"""Bounded Kurux Shackelford checkpoint from directly attested DEDR Kurux forms."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parents[2]
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 1030
candidates = {x["index"]: x for x in json.loads((P / "dedr-kurux-form-candidates.json").read_text())}
mapping = {}

def add(indices, number):
    for index in indices:
        assert index not in mapping, index
        mapping[index] = f"d{number}"

# Each mapping below is checked against a source-language DEDR form and sense.
add([9], 1977)
add([10], 5024)
add([15], 3064)
add(range(24, 28), 2244)
add(range(28, 32), 1143)
add(range(32, 36), 2023)
add([47, 49, 50], 1943)
add([60, 61, 63], 644)
add(range(115, 119), 5095)
add(range(121, 125), 1521)
add(range(169, 173), 1376)
add(range(177, 181), 1514)
add(range(189, 193), 1958)
add([197], 4711)
add([209], 4345)
add([218], 1353)
add(range(230, 234), 3271)
add(range(234, 238), 1936)
add([238], 4679)
add([281, 283], 3992)
add(range(288, 291), 3746)
add(range(318, 320), 1517)
add(range(320, 324), 5490)
add([324], 3752)
add(range(329, 331), 1040)
add(range(339, 343), 941)
add(range(359, 363), 1222)
add(range(363, 367), 399)
add([367, 368, 370], 400)
add([369], 4944)
add(range(371, 375), 1411)
add([412, 415, 417, 419], 50)
add(range(443, 447), 4813)
add(range(500, 508), 3999)
add(range(508, 512), 4275)
add([524, 525], 1458)
add([530], 4259)
add([545, 546], 1568)
add([584], 2807)
add(range(649, 653), 3263)
add(range(657, 661), 410)
add(range(661, 665), 1)
add([680], 4841)
add(range(681, 685), 1619)
add(range(693, 697), 5151)
add([710, 711, 713], 5151)
add(range(718, 722), 5151)
add([726, 728], 1)
add(range(730, 734), 410)
add(range(734, 738), 1)
add([770, 771], 976)
add(range(860, 864), 254)
add(range(868, 872), 2494)
add(range(885, 888), 3323)

prior_only = {329, 330, 363, 364, 365, 366}
for index, pid in mapping.items():
    assert 1 <= index <= len(records)
    if index not in prior_only:
        assert any(pid in match[2] for match in candidates[index]["matches"]), (index, pid)

holds = {
    7: "cɛ̃hɛra 'face' resembles an Indo-Aryan form; immediate donor into Kurux is unresolved.",
    8: "cɛ̃hɛra 'face' resembles an Indo-Aryan form; immediate donor into Kurux is unresolved.",
    62: "pīsab 'urine' is contact-shaped; the immediate donor is unresolved.",
    291: "carbī 'fat' is contact-shaped and not the DEDR 3746 nē̃tā family.",
    351: "The DEDR Kurux naṛgā 'bug' has no compiled DEDR parent node; do not assign its form ID as an etymon.",
    352: "The DEDR Kurux naṛgā 'bug' has no compiled DEDR parent node; do not assign its form ID as an etymon.",
    353: "The DEDR Kurux naṛgā 'bug' has no compiled DEDR parent node; do not assign its form ID as an etymon.",
    354: "The DEDR Kurux naṛgā 'bug' has no compiled DEDR parent node; do not assign its form ID as an etymon.",
    516: "mabɛ̄s 'bad' contains overt negation of 'good'; whole-form component analysis is deferred.",
    517: "mabɛss 'bad' contains overt negation of 'good'; whole-form component analysis is deferred.",
    518: "bɛsmalla 'bad' contains overt negation of 'good'; whole-form component analysis is deferred.",
    519: "bɛsmalla 'bad' contains overt negation of 'good'; whole-form component analysis is deferred.",
}
assert not set(mapping) & set(holds)

parents = set(mapping.values())
articles = {}
with gzip.open(R / "data/dedr/entry-texts-audit.csv.gz", "rt") as fh:
    for row in csv.DictReader(fh):
        if row["Entry_ID"] in parents:
            assert row["Entry_ID"] not in articles, row["Entry_ID"]
            articles[row["Entry_ID"]] = {k: row[k] for k in ("Page", "Citation", "Raw_Article")}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents

def kurux_excerpt(pid):
    raw = articles[pid]["Raw_Article"]
    match = re.search(r"\bKur\.\s", raw)
    assert match, pid
    rest = raw[match.end():]
    end = re.search(r"\s(?:Malt|Br|Cf|DED|\?|/|Ta|Ma|Ko|To|Ka|Koḍ|Tu|Te|Kol|Nk|Pa|Ga|Go|Konḍa|Pe|Manḍ|Kui|Kuwi)\.\s", rest)
    return (rest[:end.start()] if end else rest).strip()

def short(s):
    return s if len(s) <= 280 else s[:280].rsplit(" ", 1)[0] + "…"

accepted = []
research_holds = []
unexamined = []
for index, record in enumerate(records, 1):
    if index in mapping:
        pid = mapping[index]
        matches = [m for m in candidates.get(index, {}).get("matches", []) if pid in m[2]]
        if 504 <= index <= 507:
            matches.sort(key=lambda m: 0 if m[0] == "paccā" else 1)
        if matches:
            form, gloss, _, _ = matches[0]
            comparison = f"DEDR Kurux {form} ‘{gloss}’"
        else:
            comparison = "the same-form Kurux comparison in an older pending survey pass"
        evidence = (f"The full DEDR {pid[1:]} article gives Kurux: {short(kurux_excerpt(pid))}. "
                    f"Shackelford records {record['Form']} ‘{record['Gloss']}’; {comparison}. ")
        accepted.append(dict(record=record, parent=pid, kind="reflex",
                             citation=record["Source"] + f";dedr[entry {pid[1:]}]",
                             evidence=evidence, status="pending-review", surveyIndex=index))
    elif index in holds:
        research_holds.append(dict(record=record, reason=holds[index],
                                   status="research-hold", surveyIndex=index))
    else:
        unexamined.append(dict(record=record, reason="Outside this bounded DEDR Kurux primary-article checkpoint; exact-form search results are retained for subsequent research.",
                               status="unexamined", surveyIndex=index))
assert len(accepted) + len(research_holds) + len(unexamined) == len(records)
assert len({x["record"]["ID"] for x in accepted + research_holds + unexamined}) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted,
                                                            held=research_holds), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(research_holds, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text(json.dumps(unexamined, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
print(dict(proposed=len(accepted), holds=len(research_holds), unexamined=len(unexamined), parents=len(parents)))
