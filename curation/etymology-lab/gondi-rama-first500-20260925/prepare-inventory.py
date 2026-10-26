"""Stage a bounded, exact-ID Gondi cognate survey pass without prior/disputed cases."""
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
compiled = [r for r in csv.DictReader((DATA / "cldf/forms.csv").open())
            if r["Language_ID"] == "Gondi" and r["Source"].startswith(SOURCE + "[")
            and r["Status"] not in {"entry", "redirect"}]
forms = {r["ID"]: r for r in compiled}
assert len(forms) == len(compiled) == 10264
linked = {r["Child_ID"] for r in csv.DictReader((DATA / "cldf/edges.csv").open())
          if r["Rank"] == "1" and r["Child_ID"] in forms}
linked.update(r["Form_ID"] for r in read_assignments()
              if str(r.get("Rank")) == "1" and r["Form_ID"] in forms)
pending = set()
for filename in glob.glob(str(P.parent / "Gondi/batch-*.json")):
    manifest = json.loads(Path(filename).read_text())
    if manifest.get("status") != "pending-review":
        continue
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
assert len(residual) == 4874
excluded_glosses = {"ant", "meat", "broom", "south-broom", "south broom"}
eligible = [r for r in residual if r["Gloss"].strip().lower() not in excluded_glosses]
selected, deferred = eligible[:500], eligible[500:]
deferred += [r for r in residual if r["Gloss"].strip().lower() in excluded_glosses]
assert len(selected) == 500 and len(deferred) == 4374
assert not any(r["Gloss"].strip().lower() in excluded_glosses for r in selected)
assert len(linked | pending | held | {r["ID"] for r in selected + deferred}) == len(forms)
assert not ({r["ID"] for r in selected} & {r["ID"] for r in deferred})
(P / "survey-inventory.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
(P / "deferred-inventory.json").write_text(json.dumps(deferred, ensure_ascii=False, indent=2) + "\n")
(P / "ownership-baseline.json").write_text(json.dumps({
    "source": SOURCE, "language": "Gondi", "sourceRecords": len(forms),
    "linked": len(linked), "priorPending": len(pending), "priorHeld": len(held),
    "residual": len(residual), "selected": len(selected), "deferred": len(deferred),
    "excludedGlosses": sorted(excluded_glosses),
    "priorPendingIds": sorted(pending), "priorHeldIds": sorted(held),
    "selectedIds": [r["ID"] for r in selected], "deferredIds": [r["ID"] for r in deferred],
}, ensure_ascii=False, indent=2) + "\n")
for start in range(0, len(selected), 35):
    rows = selected[start:start + 35]
    requests = [{"mode": "lexicon", "languageId": "Gondi",
                 "params": {"form": r["Form"], "relaxed": True}} for r in rows]
    (P / f"discovery-{start+1:03d}-{start+len(rows):03d}-requests.json").write_text(
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source": len(forms), "linked": len(linked),
                  "priorPending": len(pending), "priorHeld": len(held),
                  "selected": len(selected), "deferred": len(deferred)}))
