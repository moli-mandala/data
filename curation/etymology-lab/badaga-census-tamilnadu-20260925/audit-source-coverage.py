"""Independently prove one research owner for every Badaga Census Tamil Nadu record."""
import csv
import glob
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
SOURCE = "census2023tamilnadu"
compiled = {}
with (DATA / "cldf/forms.csv").open() as stream:
    for row in csv.DictReader(stream):
        if row["Language_ID"] == "Badaga" and row["Source"].startswith(SOURCE + "["):
            assert row["ID"] not in compiled
            compiled[row["ID"]] = row
assert len(compiled) == 498
owners = {}
for filename in glob.glob(str(P.parent / "Badaga/batch-*.json")):
    manifest = json.loads(Path(filename).read_text())
    if manifest.get("researchDirectory") == str(P):
        continue
    for proposal in manifest.get("proposals", []):
        for form_id in proposal.get("formIds", []):
            if form_id in compiled:
                assert form_id not in owners, (filename, form_id)
                owners[form_id] = "previousPending"
for filename in glob.glob(str(P.parent / "*/proposed-decisions*.json")):
    if Path(filename).parent == P:
        continue
    data = json.loads(Path(filename).read_text())
    for item in data.get("held", []):
        form_id = item["record"]["ID"]
        if form_id in compiled:
            assert form_id not in owners, (filename, form_id)
            owners[form_id] = "previousHeld"

current = json.loads((P / "proposed-decisions.json").read_text())
for kind, items in [("newPending", current["accepted"]), ("newHeld", current["held"])]:
    for item in items:
        form_id = item["record"]["ID"]
        assert form_id in compiled and form_id not in owners, form_id
        assert item["record"] == compiled[form_id]
        owners[form_id] = kind
assert set(owners) == set(compiled), sorted(set(compiled) - set(owners))
counts = {kind: sum(value == kind for value in owners.values()) for kind in
          ["previousPending", "previousHeld", "newPending", "newHeld"]}
assert counts == {"previousPending": 75, "previousHeld": 2,
                  "newPending": len(current["accepted"]), "newHeld": len(current["held"])}
assert counts["newPending"] + counts["newHeld"] == 421
result = {"source": SOURCE, "language": "Badaga", "compiledSourceRecords": len(compiled),
          **{key + "Records": value for key, value in counts.items()},
          "unexaminedRecords": 0, "uniqueOwnership": True,
          "recordOwnership": dict(sorted(owners.items()))}
(P / "source-coverage-001.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"compiled": len(compiled), **counts, "unexamined": 0}))
