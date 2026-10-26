"""Bounded Kurux Kim checkpoint from directly attested DEDR Kurux forms."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parents[2]
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 1076
candidates = {x["index"]: x for x in json.loads((P / "dedr-kurux-form-candidates.json").read_text())}
cross_candidates = {x["index"]: x for x in json.loads((P / "shackelford-form-candidates.json").read_text())}
mapping = {}

def add(indices, number):
    for index in indices:
        assert index not in mapping, index
        mapping[index] = f"d{number}"

# Each mapping below is checked against a source-language DEDR form and sense.
add([2], 5479)
add([35], 4567)
add(range(38, 42), 187)
add(range(43, 47), 1958)
add([87], 677)
add(range(96, 100), 1936)
add(range(100, 103), 3271)
add([105], 4679)
add(range(140, 144), 4711)
add([199], 4428)
add(range(242, 247), 3752)
add([257, 258], 5023)
add(range(261, 265), 5490)
add(range(277, 281), 1517)
add(range(281, 285), 941)
add([324], 1630)
add(range(330, 334), 1996)
add(range(335, 338), 1159)
add(range(338, 342), 5024)
add(range(358, 361), 2023)
add([373], 4990)
add(range(398, 402), 399)
add(range(483, 487), 4719)
add(range(496, 500), 5095)
add([528, 531, 532], 1514)
add(range(562, 566), 5270)
add(range(590, 594), 2236)
add(range(612, 616), 1416)
add(range(638, 641), 254)
add(range(655, 659), 1407)
add([676, 678, 680], 3612)
add(range(684, 687), 2867)
add(range(731, 735), 5459)
add(range(889, 893), 3682)
add(range(911, 914), 3215)
add(range(915, 918), 1135)
add(range(961, 965), 1568)
add([985, 987], 3263)
add(range(1020, 1024), 5151)
add(range(1028, 1033), 410)
add(range(1033, 1038), 1)
add([1039, 1040], 410)

cross_only = set(range(43, 47)) | set(range(398, 402))
for index, pid in mapping.items():
    assert 1 <= index <= len(records)
    if index not in cross_only:
        assert any(pid in match[2] for match in candidates[index]["matches"]), (index, pid)

holds = {
    20: "The rainbow response is a compound; the whole construction needs ordered-component evidence.",
    21: "The rainbow response is a compound; the whole construction needs ordered-component evidence.",
    22: "The rainbow response is a compound; the whole construction needs ordered-component evidence.",
    23: "The rainbow response is a compound; the whole construction needs ordered-component evidence.",
    24: "The rainbow response is a compound; the whole construction needs ordered-component evidence.",
    363: "taroa 'palm' resembles the palm-tree form in DEDR 3180, but the anatomical sense is unsupported there.",
    520: "DEDR 1727 gives a pellet-bow; Kim's generic 'bow' needs the implement subtype confirmed.",
    521: "DEDR 1727 gives a pellet-bow; Kim's generic 'bow' needs the implement subtype confirmed.",
    951: "DEDR 3045 marks the Kurux wet comparison uncertain, so the family remains held.",
    952: "DEDR 3045 marks the Kurux wet comparison uncertain, so the family remains held.",
    953: "DEDR 3045 marks the Kurux wet comparison uncertain, so the family remains held.",
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
        if matches:
            form, gloss, _, _ = matches[0]
            comparison = f"DEDR Kurux {form} ‘{gloss}’"
        else:
            cross = next(m for m in cross_candidates[index]["matches"] if m[2] == pid)
            comparison = f"Shackelford Kurux {cross[0]} ‘{cross[1]}’ in an independently pending pass"
        evidence = (f"The full DEDR {pid[1:]} article gives Kurux: {short(kurux_excerpt(pid))}. "
                    f"Kim records {record['Form']} ‘{record['Gloss']}’; {comparison}. ")
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
