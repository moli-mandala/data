"""Prepare pinned Kani installation assets under the source package only."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "whole-recovery"
DATA = next(p for p in HERE.parents if (p / "tags.py").exists())
STEM = "20260927-nair-kani"
PINS = {
    "review-candidate.csv": "008927fd95809d423ac9d8b913ff690c4c398fee0da940fe96838117caeb0e85",
    "review-audit.jsonl": "798ca2ae93e9783a1a6313786ba4a4e530a0d044a3e92df29d78e30c57414582",
    "physical-units-first-reading.jsonl": "8e95ee28fe33b393477f9a2474f3e348ba1a1d6799302982e04146bb5f9023e3",
    "independent-sample/independent-original-review-pass1.json": "d7ab126926c49cd6e27e96f4b467cc4f1e23b6e66f384d8f618395b0b985ef4f",
    "independent-sample/independent-original-review-pass2-addendum.json": "c1c659a5caaa5c79df4aa3c2de4de20999f6c8ee0bc03dd1476826f621665a4e",
}


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    assert not (HERE / "installation-receipt.json").exists(), "Do not rebase an installed approval"
    for name, pin in PINS.items():
        assert sha(REVIEW / name) == pin, name
    first = json.loads((REVIEW / "independent-sample/independent-original-review-pass1.json").read_text())
    second = json.loads((REVIEW / "independent-sample/independent-original-review-pass2-addendum.json").read_text())
    assert first["sampled_source_units"] == 20 and first["sampled_actual_rows"] == 21
    assert first["sampled_material_errors"] == 0
    assert second["new_csv_sha256"] == PINS["review-candidate.csv"]
    assert second["new_audit_sha256"] == PINS["review-audit.jsonl"]
    assert second["control_result"] == "pass"
    assert second["unchanged_selected_source_units"] == 20
    audit = [json.loads(line) for line in (REVIEW / "review-audit.jsonl").read_text().splitlines()]
    accepted = []
    for raw in audit:
        entry_keys = raw["emitted_entry_keys"]
        witnesses = [{**w, "status": "ingested" if w["entry_key"] in entry_keys else "excluded_control_or_documentary"} for w in raw["witnesses"]]
        accepted.append({
            "source_unit_key": raw["source_unit_key"], "printed_page": raw["printed_page"],
            "entry_keys": entry_keys, "status": "ingested" if entry_keys else "audit_only",
            "accepted_for_source_stage": True, "witnesses": witnesses,
            "source_claim": raw["source_claim"], "note": raw["note"],
            "acceptance_evidence": "Independent 20-unit/21-row pass plus exact one-row source-literal correction and independent targeted addendum; root delegated guarded source-stage installation.",
            "source_record": raw,
        })
    assert len(accepted) == 42 and len({u["source_unit_key"] for u in accepted}) == 42
    assert sum(len(u["witnesses"]) for u in accepted) == 66
    assert sum(len(u["entry_keys"]) for u in accepted) == 36
    (HERE / "accepted-audit-proposal.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in accepted))
    settings = yaml.safe_load((REVIEW / "proposed-settings.yaml").read_text())
    settings["defaults"]["importer"] = {
        "commands": [["data/other/forms/raw_data/nair_kani_2013/import_source.py", "--install"]],
        "note": "Guarded complete Nair 2013 Kerala Kani chapter; controls remain in accepted audit.",
    }
    settings["sources"]["nair2013kani"].update({
        "identity": {"dedupe_by_entry_key": True}, "forms": {"split_alternates": False},
        "gloss": {"grammar_tags": False},
    })
    (HERE / (STEM + ".yaml")).write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    bib = (REVIEW / "proposed-references.bib").read_text().replace(
        "Complete Kani chapter, printed pages 32--33; Malayalam and Tamil Nadu Kani controls remain in the source audit",
        "Complete Kerala Kani chapter, printed pages 32--33: 36 target rows, 42 audited source units and 66 printed witnesses; 30 control or documentary witnesses remain audit-only",
    ).replace(
        "data/other/forms/raw_data/nair_kani_2013/whole-recovery/physical-units-first-reading.jsonl; data/other/forms/raw_data/nair_kani_2013/whole-recovery/review-audit.jsonl",
        "data/other/forms/20260927-nair-kani.csv; data/other/forms/raw_data/nair_kani_2013/installed-audit.jsonl",
    )
    (HERE / "proposed-references.bib").write_text(bib)
    identity = json.loads((REVIEW / "identity-and-profile-proposal.json").read_text())
    for name, fields, row in [
        ("proposed-language.csv", ["ID", "Name", "Glottocode", "Latitude", "Longitude", "Clade", "Location", "Quality"], identity["proposed_base_language"]),
        ("proposed-dialect.csv", ["ID", "Tag", "Language_ID", "Source_Language_ID", "Name", "Glottocode", "Latitude", "Longitude", "Clade", "Location", "Quality"], identity["proposed_source_variety"]),
    ]:
        with (HERE / name).open("w", newline="") as out:
            writer = csv.writer(out); writer.writerow(fields); writer.writerow([row[f] for f in fields])
    files = [*[(REVIEW / name) for name in PINS], REVIEW / "prepare_first_reading.py", REVIEW / "prepare_review_candidate.py", REVIEW / "regeneration-check.json", REVIEW / "verify_regeneration.py", REVIEW / "actual-parser-review.json", REVIEW / "identity-and-profile-proposal.json", REVIEW / "revision2-continuity.json", HERE / "accepted-audit-proposal.jsonl", HERE / (STEM + ".yaml"), HERE / "proposed-references.bib", HERE / "proposed-language.csv", HERE / "proposed-dialect.csv", HERE / "import_source.py"]
    targets = [DATA / f"data/other/forms/{STEM}.csv", DATA / f"data/other/forms/{STEM}.yaml", HERE / "installed-audit.jsonl", DATA / "cldf/languages.csv", DATA / "cldf/dialects.csv", DATA / "cldf/sources.bib"]
    assert not any(p.exists() for p in targets[:3])
    approval = {
        "installation_approved": True, "result": "pass", "source_stage_only": True,
        "authorization": "Root delegated guarded canonical Kani source-stage installation following independent sample and corrected-control addendum on 2026-09-27.",
        "rows": 36, "audit_units": 42, "witness_slots": 66,
        "files": {str(p.relative_to(HERE)): sha(p) for p in files},
        "canonical_baselines": {str(p.relative_to(DATA)): sha(p) for p in targets},
        "durable_identity_sha256": sha(DATA / "data/form-identities.csv"),
        "source_pdf_sha256": "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af",
        "deferred": ["full CLDF build", "full test suite", "compiled source-key/ID/graph checks", "browser database and QA", "publication"],
    }
    (HERE / "source-stage-authorization.json").write_text(json.dumps(approval, indent=2) + "\n")
    print("Prepared six-target source-stage plan; canonical files unchanged")


if __name__ == "__main__":
    main()
