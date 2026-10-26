"""Verify the precise source-literal control repair after failed historical freeze."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OLD = PKG / "revision1"
KEY = "nair2013kani:p33:negative:0:w1"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def keyed(path):
    return {r[10]: r for r in csv.reader(path.open(newline=""))}

old, new = keyed(OLD / "review-candidate.csv"), keyed(PKG / "review-candidate.csv")
assert old.keys() == new.keys()
assert [(k, [(i, a, b) for i, (a, b) in enumerate(zip(old[k], new[k])) if a != b])
        for k in old if old[k] != new[k]] == [(KEY, [(2, "-aa", "aa")])]
assert new[KEY][14] == old[KEY][14] and "suffix neg" in new[KEY][14]
for k in json.loads((HERE / "selection.json").read_text())["selected"]:
    for r in k["actual_rows"]:
        assert r["entry_key"] != KEY and new[r["entry_key"]] == r["columns"]

report = {
    "status": "pass; previous 20-unit original audit retained and failed edge control repaired",
    "historical_report_sha256": sha(HERE / "independent-original-review-pass1.json"),
    "old_csv_sha256": sha(OLD / "review-candidate.csv"),
    "new_csv_sha256": sha(PKG / "review-candidate.csv"),
    "new_audit_sha256": sha(PKG / "review-audit.jsonl"),
    "continuity_sha256": sha(PKG / "revision2-continuity.json"),
    "changed_entry_key": KEY,
    "old_form": "-aa", "new_form": "aa",
    "source_evidence": "Nair 2013 p. 33 original image says 'The aa negative suffix'; plain aa is printed with no leading hyphen. The source explicitly calls it a suffix, supporting unchanged suffix/neg tags.",
    "unchanged_selected_source_units": 20,
    "unchanged_selected_actual_rows": 21,
    "selected_previous_original_result": "20/20 pass; no selected lexical row changed",
    "control_result": "pass",
    "independent_new_full_sample": "not rerun; targeted one-field delta and unchanged prior selected rows proven",
    "canonical_changes": False,
}
(HERE / "independent-original-review-pass2-addendum.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(report["status"])
