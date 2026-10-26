"""Render pending Kui LSI checkpoint; do not write accepted etymology sidecars."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / "Kui"
decision = json.loads((P / "proposed-decisions.json").read_text())
articles = json.loads((P / "primary-evidence.json").read_text())
accepted, held = decision["accepted"], decision["held"]
assert decision["status"] == "pending-review"
ids = {x["parent"] for x in accepted}
parents = {r["ID"]: r for r in csv.DictReader((DATA / "cldf/forms.csv").open()) if r["ID"] in ids}
assert set(parents) == ids
path = DEST / "batch-006.json"
if path.exists():
    prior = json.loads(path.read_text())
    assert prior.get("researchDirectory") == str(P)
props = []
lines = ["**Next 3 proposals, ready for review. Nothing in this batch saved yet.**", "",
         "Kui · LSI 1928 residual records · batch 006. Pending review.", "",
         "| # | Kui | Proposed etymology | Evidence |", "|---|---|---|---|"]
for number, item in enumerate(accepted, 1):
    parent, record = item["parent"], item["record"]
    article = articles[parent]
    url = f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={article["Page"]}'
    lines.append(f'| {number} | **{record["Form"]}** ‘{record["Gloss"]}’ | Proposed reflex of [**DEDR {parent[1:]}**]({url}) | {item["evidence"]} |')
    assignment = dict(Form_ID=record["ID"], Etymon_ID=parent, Kind="reflex", Rank="1", Status="proposed",
                      Source=item["citation"], Notes=item["evidence"], Pos="")
    props.append(dict(number=number, status="pending-review", parentId=parent, parentForm=parents[parent]["Form"],
                      kind="reflex", citation=item["citation"], evidence=item["evidence"], formIds=[record["ID"]],
                      records=[record], assignments=[assignment]))
lines += ["", "Proposed assignment rows: **3**. Affected records: **3**. Relation kind: **reflex**.", "",
          f"Exact persistent IDs and complete records: [{path.name}]({path}).", "",
          "The 47 other examined forms remain in [research holds](../kui-lsi-20260925/research-holds.json); 101 are still unexamined. The preceding Kui batch 005 remains pending and unsaved."]
path.write_text(json.dumps(dict(language="Kui", batch=6, status="pending-review", date="2026-09-25",
                                researchDirectory=str(P), proposals=props), ensure_ascii=False, indent=2) + "\n")
review = path.with_name("batch-006-review.md")
review.write_text("\n".join(lines) + "\n")
(P / "review-manifest-paths.json").write_text(json.dumps({"Kui": [str(path)]}, indent=2) + "\n")

allrows = json.loads((P / "survey-inventory.json").read_text())
prior_p = json.loads((P / "previously-proposed-records.json").read_text())
prior_h = json.loads((P / "previously-held-records.json").read_text())
owned_ids = [x["record"]["ID"] for x in prior_p + prior_h + accepted + held]
assert len(owned_ids) == len(set(owned_ids)) == 74
unexamined = [r for r in allrows if r["ID"] not in set(owned_ids)]
assert len(allrows) == 175 and len(unexamined) == 101
(P / "unexamined-inventory.json").write_text(json.dumps(unexamined, ensure_ascii=False, indent=2) + "\n")
(P / "record-coverage-001.json").write_text(json.dumps(dict(sourceRecords=len(allrows), earlierProposed=len(prior_p),
    earlierHeld=len(prior_h), newProposed=len(accepted), newHeld=len(held), unexamined=len(unexamined), uniqueOwnership=True), indent=2) + "\n")
index = ["# Kui LSI 1928 residual research — checkpoint 001", "", "**Pending review. No new assignments saved. No rebuild run.**", "",
         f"Of 175 source records, {len(prior_p)} have earlier pending proposals, {len(prior_h)} an earlier hold, "
         f"{len(accepted)} have new pending proposals, {len(held)} new holds, and {len(unexamined)} remain unexamined.", "",
         f"Review batch: [006]({review}).", "",
         "[Source records](source-record-evidence.json), [homepage form search](discovery-001-results.json), "
         "[homepage gloss search](followup-001-results.json), [full DEDR articles](primary-evidence.json), "
         "[decisions](proposed-decisions.json), and [holds](research-holds.json) preserve the research basis.", "",
         "Animal and descriptive phrases were held where the whole expression lacks a complete component analysis. "
         "No source text, parent heading, or accepted sidecar was edited."]
(P / "REVIEW.md").write_text("\n".join(index) + "\n")
print(json.dumps(dict(proposed=len(accepted), held=len(held), unexamined=len(unexamined), batch=6)))
