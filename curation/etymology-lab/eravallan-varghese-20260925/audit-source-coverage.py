"""Prove every compiled Eravallan Palakkad record has one research disposition."""
import csv
import glob
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
SOURCE = "varghese2015palakkad"
compiled = {}
with (DATA / "cldf/forms.csv").open() as fh:
    for row in csv.DictReader(fh):
        if row["Language_ID"] == "Eravallan" and row["Source"].startswith(SOURCE + "["):
            assert row["ID"] not in compiled
            compiled[row["ID"]] = row

owners = {}
for path in glob.glob(str(P.parent / "Eravallan/batch-*.json")):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("researchDirectory") == str(P):
        continue
    for proposal in manifest.get("proposals", []):
        for form_id in proposal.get("formIds", []):
            if form_id in compiled:
                assert form_id not in owners, (path, form_id)
                owners[form_id] = "previousPending"

earlier = json.loads((P.parent / "dravidian-surveys-20260921/proposed-decisions.json").read_text())
for item in earlier["held"]:
    form_id = item["record"]["ID"]
    if form_id in compiled:
        assert form_id not in owners, form_id
        owners[form_id] = "previousHeld"

current = json.loads((P / "proposed-decisions.json").read_text())
for kind, items in [("newPending", current["accepted"]), ("newHeld", current["held"])]:
    for item in items:
        form_id = item["record"]["ID"]
        assert form_id in compiled and form_id not in owners, form_id
        owners[form_id] = kind

assert set(owners) == set(compiled), sorted(set(compiled) - set(owners))
counts = {kind: sum(v == kind for v in owners.values()) for kind in
          ["previousPending", "previousHeld", "newPending", "newHeld"]}
assert counts == {"previousPending": 297, "previousHeld": 17,
                  "newPending": 70, "newHeld": 70}, counts
result = {"source": SOURCE, "language": "Eravallan",
          "compiledSourceRecords": len(compiled), **{k + "Records": v for k, v in counts.items()},
          "unexaminedRecords": 0, "uniqueOwnership": True,
          "recordOwnership": dict(sorted(owners.items()))}
(P / "source-coverage-001.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"compiled": len(compiled), **counts, "unexamined": 0}))
