"""Compare Toda pending manifests with the canonical read-only dry run."""
import collections
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[2]))
from etymology_lab import rows_from_decisions

checkpoint = sys.argv[1]
log = (P / f"check-pass-{checkpoint}.log").read_text()
validation = json.JSONDecoder().raw_decode(log[log.index("{"):])[0]
assert validation["dryRun"] and validation["secondApplicationChanges"] == 0
decisions = json.loads((P / "proposed-decisions.json").read_text())
expected = rows_from_decisions(decisions["accepted"], "")
actual = []
for filename in json.loads((P / "review-manifest-paths.json").read_text()):
    manifest = json.loads(Path(filename).read_text())
    assert manifest["status"] == "pending-review"
    assert manifest["researchDirectory"] == str(P)
    for proposal in manifest["proposals"]:
        assert proposal["status"] == "pending-review"
        actual.extend(proposal["assignments"])
key = lambda r: tuple(str(r.get(k, "") or "") for k in ["Form_ID", "Etymon_ID", "Kind", "Rank", "Pos"])
assert collections.Counter(map(key, actual)) == collections.Counter(map(key, expected))
assert len(actual) == validation["assignmentRows"]
coverage = json.loads((P / "source-coverage-001.json").read_text())
assert coverage["uniqueOwnership"] and coverage["proposedRecords"] == validation["affectedRecords"]
assert coverage["linkedRecords"] + coverage["priorProposed"] + coverage["priorHeld"] + coverage["proposedRecords"] + coverage["heldRecords"] + coverage["unexaminedRecords"] == coverage["compiledSourceRecords"] == 8879
(P / f"checkpoint-{checkpoint}-validation.json").write_text(json.dumps({"manifestAgreement": True,
    "coverageAgreement": True, "assignmentRows": len(actual), "validation": validation}, indent=2) + "\n")
print(json.dumps({"checkpoint": checkpoint, "manifestAgreement": True, "coverageAgreement": True,
                  "records": validation["affectedRecords"], "rows": len(actual),
                  "secondApplicationChanges": validation["secondApplicationChanges"]}))
