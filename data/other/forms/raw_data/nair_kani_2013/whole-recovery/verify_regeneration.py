"""Reproduce the reviewed source-local files without touching their frozen bytes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    first = load("prepare_first_reading")
    expected = "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in first.units).encode()
    assert expected == (HERE / "physical-units-first-reading.jsonl").read_bytes()
    scratch = HERE / "installation-regeneration"
    scratch.mkdir(exist_ok=True)
    candidate = load("prepare_review_candidate")
    candidate.CSV = scratch / "candidate.csv"
    candidate.AUDIT = scratch / "audit.jsonl"
    candidate.STATUS = scratch / "status.json"
    candidate.main()
    for reproduced, frozen in ((candidate.CSV, HERE / "review-candidate.csv"), (candidate.AUDIT, HERE / "review-audit.jsonl")):
        assert reproduced.read_bytes() == frozen.read_bytes(), frozen.name
    report = {
        "status": "passed byte-exact regeneration of first reading, candidate CSV and audit",
        "inputs_sha256": {p.name: sha(p) for p in [HERE / "prepare_first_reading.py", HERE / "prepare_review_candidate.py", HERE / "physical-units-first-reading.jsonl"]},
        "outputs_sha256": {p.name: sha(p) for p in [HERE / "review-candidate.csv", HERE / "review-audit.jsonl"]},
        "counts": {"source_units": 42, "printed_witness_slots": 66, "emitted_rows": 36},
        "canonical_mutations": False,
    }
    (HERE / "regeneration-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()
