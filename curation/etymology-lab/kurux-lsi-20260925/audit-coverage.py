"""Reconcile every LSI Kurux source identity without changing graph state."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
SOURCE = P.parent / "dravidian-surveys-20260921" / "proposed-decisions.json"
all_source = [r for r in csv.DictReader(open(ROOT / "cldf/forms.csv"))
              if r["Language_ID"] == "Kurux" and "LSI-KURUXORORAO" in r["Tags"]]
residual = json.loads((P / "survey-inventory.json").read_text())
decisions = json.loads((P / "proposed-decisions.json").read_text())
earlier = json.loads(SOURCE.read_text())
ids = {r["ID"] for r in all_source}
new_prop = {x["record"]["ID"] for x in decisions["accepted"]}
new_hold = {x["record"]["ID"] for x in decisions["held"]}
prior_prop = {x["record"]["ID"] for x in earlier["accepted"] if x["record"]["ID"] in ids}
prior_hold = {x["record"]["ID"] for x in earlier["held"] if x["record"]["ID"] in ids}
cats = [new_prop, new_hold, prior_prop, prior_hold]
assert all(a.isdisjoint(b) for j,a in enumerate(cats) for b in cats[j+1:])
assert ids == set.union(*cats)
assert {r["ID"] for r in residual} == new_prop | new_hold
rec_by_id = {r["ID"]: r for r in all_source}
external = [rec_by_id[i] for i in sorted(prior_prop | prior_hold)]
(P / "externally-assigned-records.json").write_text(json.dumps(external, ensure_ascii=False, indent=2) + "\n")
(P / "previously-proposed-records.json").write_text(json.dumps([rec_by_id[i] for i in sorted(prior_prop)], ensure_ascii=False, indent=2) + "\n")
(P / "previously-held-records.json").write_text(json.dumps([rec_by_id[i] for i in sorted(prior_hold)], ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
count = {"source": len(ids), "residual": len(residual), "previouslyProposed": len(prior_prop),
         "previouslyHeld": len(prior_hold), "newProposedRecords": len(new_prop),
         "newProposedRows": sum(len(x.get("components", [x["parent"]])) for x in decisions["accepted"]),
         "newHeld": len(new_hold), "unexamined": 0}
(P / "inventory-counts.json").write_text(json.dumps(count, indent=2) + "\n")
(P / "record-coverage-001.json").write_text(json.dumps({"counts": count,
    "sourceRecordIds": sorted(ids), "newProposalIds": sorted(new_prop), "newHoldIds": sorted(new_hold),
    "priorProposalIds": sorted(prior_prop), "priorHoldIds": sorted(prior_hold)}, indent=2) + "\n")
print(json.dumps(count))
