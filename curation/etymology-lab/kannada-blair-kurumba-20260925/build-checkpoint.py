"""Build a bounded, review-only Kannada Kurumba etymology checkpoint."""
import glob
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 1504
screened = json.loads((P / "primary-screen-candidates.json").read_text())
mapping = {x["index"]: x for x in screened}
assert len(mapping) == len(screened)
all_articles = json.loads((P / "candidate-article-evidence.json").read_text())
articles = {pid: all_articles[pid] for pid in {x["parent"] for x in screened}}

searches = []
for filename in sorted(glob.glob(str(P / "discovery-*-results.json")),
                       key=lambda path: int(Path(path).name.split("-")[1])):
    searches.extend(json.loads(Path(filename).read_text()))
assert len(searches) == len(records), (len(searches), len(records))

accepted, unexamined = [], []
for index, (record, search) in enumerate(zip(records, searches), 1):
    assert search["opts"]["params"]["form"] == record["Form"]
    choice = mapping.get(index)
    if choice is None:
        unexamined.append({
            "record": record, "status": "unexamined", "surveyIndex": index,
            "reason": "Outside this bounded primary-article checkpoint; the exact-form homepage results and unresolved cross-pass candidates remain available for further review.",
        })
        continue
    assert choice["recordId"] == record["ID"]
    parent = choice["parent"]
    evidence = (
        f"The full DEDR {parent[1:]} article includes {choice['comparandum']}; "
        "the listed Blair et al. Kurumba forms and senses are proposed within that family."
    )
    accepted.append({
        "record": record, "parent": parent, "kind": "reflex",
        "citation": record["Source"] + f";dedr[entry {parent[1:]}]",
        "evidence": evidence, "status": "pending-review", "surveyIndex": index,
    })

assert len(accepted) + len(unexamined) == len(records)
owners = {x["record"]["ID"]: "proposed" for x in accepted}
owners.update({x["record"]["ID"]: "unexamined" for x in unexamined})
assert len(owners) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({
    "status": "pending-review", "accepted": accepted, "held": [],
}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text("[]\n")
(P / "unexamined-inventory.json").write_text(json.dumps(unexamined, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-coverage-001.json").write_text(json.dumps({
    "source": "blairetal2012kurumba", "language": "Kannada",
    "compiledSourceRecords": 2507, "priorProposedRecords": 941,
    "priorHeldRecords": 62, "residualRecords": 1504,
    "proposedRecords": len(accepted), "heldRecords": 0,
    "unexaminedRecords": len(unexamined), "uniqueOwnership": True,
    "recordOwnership": owners,
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"proposed": len(accepted), "held": 0,
                  "unexamined": len(unexamined), "articles": len(articles)}))
