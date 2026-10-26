"""Reconcile all Shackelford Kurux source IDs and pending review manifests."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parents[2]
source = {r["ID"]: r for r in csv.DictReader((R / "cldf/forms.csv").open())
          if r["Language_ID"] == "Kurux"
          and r["Source"].startswith("shackelford-swenson-chaudhary-maggard2022kurux[")}
assert len(source) == 1319
old = json.loads((P.parent / "dravidian-surveys-20260921/proposed-decisions.json").read_text())
old_accepted = {x["record"]["ID"] for x in old["accepted"] if x["record"]["ID"] in source}
old_held = {x["record"]["ID"] for x in old["held"] if x["record"]["ID"] in source}
data = json.loads((P / "proposed-decisions.json").read_text())
unexamined = json.loads((P / "unexamined-inventory.json").read_text())
new_accepted = {x["record"]["ID"] for x in data["accepted"]}
new_held = {x["record"]["ID"] for x in data["held"]}
new_unexamined = {x["record"]["ID"] for x in unexamined}
sets = [old_accepted, old_held, new_accepted, new_held, new_unexamined]
assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
assert set(source) == set.union(*sets)
assert [len(x) for x in sets] == [239, 50, len(data["accepted"]), len(data["held"]), len(unexamined)]
assert len(new_accepted | new_held | new_unexamined) == 1030
paths = json.loads((P / "review-manifest-paths.json").read_text())
manifest_ids = collections.Counter(
    record["ID"] for path in paths
    for proposal in json.loads(Path(path).read_text())["proposals"]
    for record in proposal["records"])
assert manifest_ids == collections.Counter(new_accepted)
rows = sum(len(proposal["assignments"]) for path in paths
           for proposal in json.loads(Path(path).read_text())["proposals"])
assert rows == len(data["accepted"])
owners = {id: "prior-proposed" for id in old_accepted}
owners.update({id: "prior-held" for id in old_held})
owners.update({id: "proposed" for id in new_accepted})
owners.update({id: "held" for id in new_held})
owners.update({id: "unexamined" for id in new_unexamined})
report = dict(source="shackelford-swenson-chaudhary-maggard2022kurux", language="Kurux",
              compiledSourceRecords=len(source), priorProposed=len(old_accepted), priorHeld=len(old_held),
              proposedRecords=len(new_accepted), heldRecords=len(new_held),
              unexaminedRecords=len(new_unexamined), pendingBatches=len(paths),
              assignmentRows=rows, uniqueOwnership=True, manifestAgreement=True,
              recordOwnership=owners)
(P / "source-coverage-001.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print({k: report[k] for k in ("compiledSourceRecords", "priorProposed", "priorHeld", "proposedRecords", "heldRecords", "unexaminedRecords", "pendingBatches")})
