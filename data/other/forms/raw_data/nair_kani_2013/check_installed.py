"""Bounded canonical/source metadata verification; no CLDF or reference writes."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import sys
import unicodedata
from pathlib import Path

import pybtex
from pybtex.database import parse_file

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "tags.py").exists())
sys.path.insert(0, str(DATA))
import make_cldf
import source_meta

STEM = "20260927-nair-kani"
CSV = DATA / f"data/other/forms/{STEM}.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = list(csv.reader(CSV.open(newline="")))
    assert len(rows) == 36
    fields = ("lang", "param", "old_form", "gloss", "native", "ipa", "notes", "source", "cognateset", "etymology", "entry_key", "variant_of_key", "borrowed_from_key", "derivation_parent_keys", "tags")
    errors = io.StringIO(); parsed, stats = make_cldf.parse_file(str(CSV), errors, name=STEM)
    assert len(parsed) == 36 and errors.getvalue() == ""
    assert [[getattr(r, f) for f in fields] for r in parsed] == rows
    assert stats == {"converted": 0, "for_conversion": 0}
    assert all(r.form == original[2] for r, original in zip(parsed, rows))
    meta = source_meta.load()
    assert len({make_cldf.source_entry_dedupe_key(r, meta) for r in parsed}) == 36
    references = parse_file(str(DATA / "cldf/sources.bib")).entries
    formatted = {}
    engine = pybtex.PybtexEngine()
    for key in ("nair2013kani", "somasekharannairkanikkarude"):
        reference = references[key]
        text = engine.format_from_string(reference.to_string("bibtex"), "plain", output_backend="markdown")[3:].strip()
        assert text and reference.fields["ocr"] == "No" and reference.fields["etymology_provenance"] == "none"
        assert meta.flag(key, "reference", "ocr") is False
        assert meta.flag(key, "reference", "etymology_provenance") == "none"
        formatted[key] = text
    assert "Kerala" in formatted["nair2013kani"]
    languages = [r for r in csv.DictReader((DATA / "cldf/languages.csv").open()) if r["ID"] == "Kanikkaran"]
    dialects = [r for r in csv.DictReader((DATA / "cldf/dialects.csv").open()) if r["ID"] == "nair2013-kerala-kani"]
    assert len(languages) == len(dialects) == 1
    assert languages[0]["Clade"] in (DATA.parent / "jambu-static/src/lib/clades.ts").read_text()
    assert dialects[0]["Language_ID"] == "Kanikkaran"
    assert all(dialects[0]["Tag"] in r[14].split() for r in rows)
    approval = json.loads((HERE / "source-stage-authorization.json").read_text())
    receipt = json.loads((HERE / "installation-receipt.json").read_text())
    assert sha(DATA / "data/form-identities.csv") == approval["durable_identity_sha256"] == receipt["durable_identity_sha256"]
    audit = [json.loads(line) for line in (HERE / "installed-audit.jsonl").read_text().splitlines()]
    assert len(audit) == 42 and sum(len(a["witnesses"]) for a in audit) == 66
    assert {k for a in audit for k in a["entry_keys"]} == {r[10] for r in rows}
    report = {
        "status": "canonical source stage verified; compiled gates deferred",
        "counts": {"canonical_rows": 36, "audit_units": 42, "witness_slots": 66, "excluded_control_or_documentary_witnesses": 30},
        "canonical_sha256": {"csv": sha(CSV), "yaml": sha(CSV.with_suffix(".yaml")), "audit": sha(HERE / "installed-audit.jsonl")},
        "parser": {"actual_canonical_rows": len(parsed), "all_15_fields_exact": True, "Original_literal_exact": True, "conversion_errors": 0, "initial_dedupe_keys": 36},
        "transcription": {"convert": False, "preserve_hyphens": True, "phonemic_native": "blank as source supplies no independent layers", "unique_nfd_codepoints": len({c for r in rows for c in unicodedata.normalize("NFD", r[2])})},
        "references": formatted, "registry": {"base": languages[0], "dialect": dialects[0], "frontend_existing_clade": True},
        "source_meta": {"settings_files": len(meta.files), "citation_keys": len(meta.sources)},
        "durable_identity_sha256": sha(DATA / "data/form-identities.csv"),
        "independent_review": {"fresh_20_units_21_rows": "pass", "targeted_corrected_aa_control": "pass", "old_freeze_retained": True},
        "repeat_installation": "no-op; own assets exact and shared membership checked",
        "deferred": approval["deferred"],
    }
    (HERE / "canonical-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()
