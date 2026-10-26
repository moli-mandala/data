"""Apply the one source-literal correction found in the independent pass-one audit."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = HERE / "independent-sample/independent-original-review-pass1.json"
OLD = HERE / "revision1"
KEY = "nair2013kani:p33:negative:0:w1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    assert sha(REPORT) == "d7ab126926c49cd6e27e96f4b467cc4f1e23b6e66f384d8f618395b0b985ef4f"
    report = json.loads(REPORT.read_text())
    assert report["input_sha256"]["physical-units-first-reading.jsonl"] == sha(OLD / "physical-units-first-reading.jsonl")
    assert report["input_sha256"]["review-candidate.csv"] == sha(OLD / "review-candidate.csv")
    assert report["input_sha256"]["review-audit.jsonl"] == sha(OLD / "review-audit.jsonl")
    for script in ("prepare_first_reading.py", "prepare_review_candidate.py", "check_candidate.py", "validate_actual_parser.py"):
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    before = lines(OLD / "physical-units-first-reading.jsonl")
    after = lines(HERE / "physical-units-first-reading.jsonl")
    assert len(before) == len(after) == 42
    unit_changes = []
    for a, b in zip(before, after):
        assert a["source_unit_key"] == b["source_unit_key"]
        if a != b:
            unit_changes.append(a["source_unit_key"])
            assert b["source_unit_key"] == "nair2013kani:p33:negative:0"
            assert a["source_witnesses"][0]["form"] == "-aa"
            assert b["source_witnesses"][0]["form"] == "aa"
            for field in a:
                if field not in {"source_claim", "source_witnesses"}:
                    assert a[field] == b[field], field
    assert unit_changes == ["nair2013kani:p33:negative:0"]
    with (OLD / "review-candidate.csv").open(newline="") as file:
        old_rows = list(csv.reader(file))
    with (HERE / "review-candidate.csv").open(newline="") as file:
        new_rows = list(csv.reader(file))
    assert len(old_rows) == len(new_rows) == 36
    csv_changes = []
    for old, new in zip(old_rows, new_rows):
        assert old[10] == new[10]
        if old != new:
            csv_changes.append(old[10])
            assert old[10] == KEY
            assert old[2] == "-aa" and new[2] == "aa"
            assert old[:2] == new[:2] and old[3:] == new[3:]
    assert csv_changes == [KEY]
    old_audit = lines(OLD / "review-audit.jsonl")
    new_audit = lines(HERE / "review-audit.jsonl")
    assert len(old_audit) == len(new_audit) == 42
    audit_changes = [a["source_unit_key"] for a, b in zip(old_audit, new_audit) if a != b]
    assert audit_changes == unit_changes
    assert [r[10] for r in old_rows] == [r[10] for r in new_rows]
    result = {
        "status": "source-literal correction applied; independent fresh output review still pending",
        "independent_report_sha256": sha(REPORT),
        "changed_source_unit_keys": unit_changes,
        "changed_entry_keys": csv_changes,
        "old_form": "-aa",
        "new_form": "aa",
        "interpretation": "Printed plain aa remains tagged suffix neg because the source calls it a negative suffix; no punctuation is supplied in Form.",
        "counts": {"physical_units": len(after), "witness_slots": 66, "candidate_rows": len(new_rows)},
        "old_sha256": {name: sha(OLD / name) for name in ("physical-units-first-reading.jsonl", "review-candidate.csv", "review-audit.jsonl")},
        "new_sha256": {name: sha(HERE / name) for name in ("physical-units-first-reading.jsonl", "review-candidate.csv", "review-audit.jsonl", "actual-parser-review.json")},
        "checks": "source-local candidate and actual parser NFC/NFD passed; all noncorrected rows exact",
    }
    (HERE / "revision2-continuity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
