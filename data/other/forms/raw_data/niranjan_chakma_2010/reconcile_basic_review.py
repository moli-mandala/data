"""Merge the complete independent review without altering frozen first readings."""
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent


def main():
    owner_path = ROOT / "basic-first-reading.jsonl"
    review_path = ROOT / "independent-basic-full-review-20260927.json"
    owner = list(map(json.loads, owner_path.read_text().splitlines()))
    review = json.loads(review_path.read_text())
    assert sha(owner_path) == review["input_sha256"][owner_path.name]
    reviewed = {r["source_unit_key"]: r for r in review["coverage"]}
    assert len(owner) == len(reviewed) == review["coverage_count"] == 99
    assert {r["source_unit_key"] for r in owner} == set(reviewed)
    corrections = {r["source_unit_key"]: r for r in review["material_corrections"]}
    merged = []
    for original in owner:
        r = reviewed[original["source_unit_key"]]
        assert r["owner_form"] == original["raw_owner_first_reading"]
        assert r["english"] == original["printed_english"]
        for field in ("pdf_page", "printed_page", "block", "row"):
            assert r[field] == original[field]
        if r["reviewed_form"] != r["owner_form"]:
            correction = corrections[r["source_unit_key"]]
            assert (correction["old"], correction["new"]) == (r["owner_form"], r["reviewed_form"])
        record = dict(original)
        record.update({
            "reviewed_reading": r["reviewed_form"],
            "bengali_control": r["bengali_control"],
            "bengali_control_scope": r["bengali_control_scope"],
            "independent_review_status": r["status"],
            "independent_review_evidence": r,
            "review_status": "independent full review reconciled; whole-book integration pending",
            "accepted_for_installation": False,
        })
        # Preserve unresolved glyphs as explicit holds, not approved spellings.
        record["transcription_hold"] = r["status"] == "unresolved_glyph"
        record["typed_remaining_uncertainty"] = r.get("typed_uncertainty", "")
        merged.append(record)
    statuses = Counter(r["independent_review_status"] for r in merged)
    assert statuses["correction"] == len(corrections) == 3
    assert sum(r["transcription_hold"] for r in merged) == 3
    assert sum(bool(r["bengali_control"]) for r in merged) == 97
    target = ROOT / "basic-reviewed-reading.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    status = {
        "status": "all99 independent reviews reconciled; three explicit transcription holds remain",
        "cells": 99, "nonblank_bengali_controls": 97, "review_status_counts": dict(statuses),
        "corrections": review["material_corrections"],
        "holds": [{"key": r["source_unit_key"], "reason": r["typed_remaining_uncertainty"]}
                  for r in merged if r["transcription_hold"]],
        "inputs": {owner_path.name: sha(owner_path), review_path.name: sha(review_path)},
        "output": {target.name: sha(target)},
        "source_semantics_policy": "Keep English, Bengali controls and source mismatches separately; no semantic emendation.",
        "installed_rows_changed": 0,
    }
    (ROOT / "basic-reviewed-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
