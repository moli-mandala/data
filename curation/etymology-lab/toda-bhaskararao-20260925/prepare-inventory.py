"""Stage the genuinely unexamined Toda Bhaskararao dictionary records for review."""
import csv
import glob
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
sys.path.insert(0, str(DATA))
from etymology_assignments import read_assignments

SOURCE = "bhaskararao-toda2025"
compiled = [r for r in csv.DictReader((DATA / "cldf/forms.csv").open())
            if r["Language_ID"] == "Toda" and r["Source"].startswith(SOURCE + "[")]
forms = {r["ID"]: r for r in compiled}
assert len(forms) == len(compiled) == 8879
linked = {r["Child_ID"] for r in csv.DictReader((DATA / "cldf/edges.csv").open())
          if r["Rank"] == "1" and r["Child_ID"] in forms}
linked.update(r["Form_ID"] for r in read_assignments()
              if str(r.get("Rank")) == "1" and r["Form_ID"] in forms)
pending = set()
for filename in glob.glob(str(P.parent / "Kota/batch-*.json")):
    manifest = json.loads(Path(filename).read_text())
    for proposal in manifest.get("proposals", []):
        pending.update(fid for fid in proposal.get("formIds", []) if fid in forms)
held = set()
for filename in glob.glob(str(P.parent / "*/proposed-decisions*.json")):
    decisions = json.loads(Path(filename).read_text())
    pending.update(x["record"]["ID"] for x in decisions.get("accepted", [])
                   if x["record"]["ID"] in forms)
    held.update(x["record"]["ID"] for x in decisions.get("held", [])
                if x["record"]["ID"] in forms)
assert not (linked & pending or linked & held or pending & held)
residual = [r for r in compiled if r["ID"] not in linked | pending | held]
assert len(residual) == 5605
assert len(linked | pending | held | {r["ID"] for r in residual}) == len(forms)
(P / "survey-inventory.json").write_text(json.dumps(residual, ensure_ascii=False, indent=2) + "\n")
(P / "ownership-baseline.json").write_text(json.dumps({
    "source": SOURCE, "language": "Toda", "sourceRecords": len(forms),
    "linked": len(linked), "priorPending": len(pending), "priorHeld": len(held),
    "residual": len(residual), "priorPendingIds": sorted(pending),
    "priorHeldIds": sorted(held), "residualIds": [r["ID"] for r in residual],
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source": len(forms), "linked": len(linked),
                  "priorPending": len(pending), "priorHeld": len(held),
                  "residual": len(residual)}))
