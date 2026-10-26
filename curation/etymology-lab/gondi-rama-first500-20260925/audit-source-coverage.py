"""Independently prove exact owner or explicit deferral for every Rama Gondi record."""
import csv
import glob
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
sys.path.insert(0, str(DATA))
from etymology_assignments import read_assignments

SOURCE = "rama-coltekin-sofroniev2017gondi"
compiled = {}
with (DATA / "cldf/forms.csv").open() as stream:
    for row in csv.DictReader(stream):
        if row["Language_ID"] == "Gondi" and row["Source"].startswith(SOURCE + "[") and row["Status"] not in {"entry", "redirect"}:
            assert row["ID"] not in compiled
            compiled[row["ID"]] = row
assert len(compiled) == 10264
owners = {}

for row in csv.DictReader((DATA / "cldf/edges.csv").open()):
    fid = row["Child_ID"]
    if row["Rank"] == "1" and fid in compiled:
        owners[fid] = "linked"
for row in read_assignments():
    fid = row["Form_ID"]
    if str(row.get("Rank")) == "1" and fid in compiled:
        assert fid not in owners or owners[fid] == "linked"
        owners[fid] = "linked"

for filename in glob.glob(str(P.parent / "Gondi/batch-*.json")):
    manifest = json.loads(Path(filename).read_text())
    if manifest.get("researchDirectory") == str(P) or manifest.get("status") != "pending-review":
        continue
    for proposal in manifest.get("proposals", []):
        for fid in proposal.get("formIds", []):
            if fid in compiled:
                assert fid not in owners, (filename, fid)
                owners[fid] = "previousPending"
for filename in glob.glob(str(P.parent / "*/proposed-decisions*.json")):
    if Path(filename).parent == P:
        continue
    decisions = json.loads(Path(filename).read_text())
    for row in decisions.get("held", []):
        fid = row["record"]["ID"]
        if fid in compiled:
            assert fid not in owners, (filename, fid)
            owners[fid] = "previousHeld"

current = json.loads((P / "proposed-decisions.json").read_text())
for category, rows in [("newPending", current["accepted"]), ("newHeld", current["held"])]:
    for row in rows:
        fid = row["record"]["ID"]
        assert fid in compiled and fid not in owners, fid
        assert row["record"] == compiled[fid]
        owners[fid] = category
deferred = json.loads((P / "deferred-inventory.json").read_text())
for row in deferred:
    fid = row["ID"]
    assert fid in compiled and fid not in owners and row == compiled[fid]
    owners[fid] = "deferredUnexamined"
assert set(owners) == set(compiled), sorted(set(compiled) - set(owners))
counts = {category: sum(value == category for value in owners.values()) for category in
          ["linked", "previousPending", "previousHeld", "newPending", "newHeld", "deferredUnexamined"]}
assert counts["linked"] == 4708 and counts["previousPending"] == 309 and counts["previousHeld"] == 373
assert counts["newPending"] + counts["newHeld"] == 500
assert counts["deferredUnexamined"] == 4374
result = {"source": SOURCE, "language": "Gondi", "compiledSourceRecords": len(compiled),
          **{key + "Records": value for key, value in counts.items()},
          "uniqueOwnershipOrDeferral": True, "recordOwnership": dict(sorted(owners.items()))}
(P / "source-coverage-001.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"compiled": len(compiled), **counts}))
