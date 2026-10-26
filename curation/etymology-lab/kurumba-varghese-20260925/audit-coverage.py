"""Read-only exact-record and review-manifest audit for the Varghese Kurumba pass."""
import collections
import csv
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
sys.path.insert(0, str(DATA))
from etymology_lab import rows_from_decisions

source = "varghese2015palakkad"
language = "Kurumba"
inventory = json.loads((P / "survey-inventory.json").read_text())
decisions = json.loads((P / "proposed-decisions.json").read_text())
current = {r["ID"]: r for r in inventory}
assert len(current) == len(inventory) == 150
new = decisions["accepted"] + decisions["held"]
assert len(new) == 150
new_by_id = {x["record"]["ID"]: x["record"] for x in new}
assert len(new_by_id) == len(new) and new_by_id == current

compiled = {}
with (DATA / "cldf/forms.csv").open() as f:
    for r in csv.DictReader(f):
        if r["Language_ID"] == language and source in r["Source"]:
            compiled[r["ID"]] = r

registry = json.loads((P.parent / "dravidian-surveys-20260921/goal-pass-registry.json").read_text())
prior = {}
owners = collections.defaultdict(list)
for dirname in registry:
    if dirname == P.name:
        continue
    path = P.parent / dirname / "proposed-decisions.json"
    if not path.exists():
        continue
    bundle = json.loads(path.read_text())
    for x in bundle.get("accepted", []) + bundle.get("held", []):
        r = x["record"]
        if r["Language_ID"] == language and source in r["Source"]:
            prior[r["ID"]] = r
            owners[r["ID"]].append(dirname)
assert all(len(v) == 1 for v in owners.values()), {k:v for k,v in owners.items() if len(v) != 1}
assert not (set(prior) & set(current))
assert set(compiled) == (set(prior) | set(current)), (len(compiled), len(prior), len(current), sorted(set(compiled) - set(prior) - set(current))[:5])
assert all(compiled[k] == v for k, v in {**prior, **current}.items())

# Old review manifests may contain proposed IDs absent from a prior decision
# bundle; inspect them directly before claiming unique review ownership.
older_manifest_overlap = []
for path in (P.parent / language).glob("batch-*.json"):
    manifest = json.loads(path.read_text())
    if manifest.get("researchDirectory") == str(P):
        continue
    previous_ids = {record_id for proposal in manifest.get("proposals", [])
                    for record_id in proposal.get("formIds", [])}
    overlap = set(current) & previous_ids
    if overlap:
        older_manifest_overlap.append((str(path), sorted(overlap)))
assert not older_manifest_overlap, older_manifest_overlap

manifest_paths = json.loads((P / "review-manifest-paths.json").read_text())[language]
manifest_rows = []
manifest_records = []
for path in manifest_paths:
    m = json.loads(Path(path).read_text())
    assert m["status"] == "pending-review" and m["language"] == language
    for proposal in m["proposals"]:
        assert proposal["status"] == "pending-review"
        manifest_rows.extend(proposal["assignments"])
        manifest_records.extend(proposal["formIds"])
assert collections.Counter(manifest_records) == collections.Counter(x["record"]["ID"] for x in decisions["accepted"])
expected = rows_from_decisions(decisions["accepted"], "")
key = lambda r: tuple(str(r.get(c, "") or "") for c in ("Form_ID", "Etymon_ID", "Kind", "Rank", "Pos"))
assert collections.Counter(map(key, manifest_rows)) == collections.Counter(map(key, expected))
assert not json.loads((P / "unexamined-inventory.json").read_text())

report = {
    "source": source,
    "language": language,
    "totalCompiledSourceRecords": len(compiled),
    "priorPendingProposalsAndHolds": len(prior),
    "newPendingProposals": len(decisions["accepted"]),
    "newProposedAssignmentRows": len(expected),
    "newResearchHolds": len(decisions["held"]),
    "unexamined": 0,
    "exactRecordPreservation": True,
    "uniquePriorPassOwnership": True,
    "olderReviewManifestOverlap": 0,
    "reviewManifestAgreement": True,
    "reviewBatches": len(manifest_paths),
}
for name in ("inventory-counts.json", "record-coverage-001.json"):
    (P / name).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
