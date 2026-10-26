"""Install the approved complete Kani chapter. Default is a read-only dry run.

A durable staged transaction permits recovery of interrupted writes. No lexical
generator, CLDF pipeline, database build, publication or remote command runs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import unicodedata
from pathlib import Path

import yaml
from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "tags.py").exists())
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta

STEM = "20260927-nair-kani"
APPROVAL = HERE / "source-stage-authorization.json"
RECEIPT = HERE / "installation-receipt.json"
JOURNAL = HERE / "installation-transaction.json"
IDENTITY = DATA / "data/form-identities.csv"
INPUTS = {"csv": "whole-recovery/review-candidate.csv", "yaml": "20260927-nair-kani.yaml", "audit": "accepted-audit-proposal.jsonl", "languages": "proposed-language.csv", "dialects": "proposed-dialect.csv", "bib": "proposed-references.bib"}
TARGETS = {"csv": DATA / f"data/other/forms/{STEM}.csv", "yaml": DATA / f"data/other/forms/{STEM}.yaml", "audit": HERE / "installed-audit.jsonl", "languages": DATA / "cldf/languages.csv", "dialects": DATA / "cldf/dialects.csv", "bib": DATA / "cldf/sources.bib"}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_hash(path: Path) -> str | None:
    return digest(path.read_bytes()) if path.exists() else None


def validate_inputs() -> dict:
    approval = json.loads(APPROVAL.read_text())
    assert approval["installation_approved"] is True and approval["result"] == "pass"
    assert set(INPUTS.values()) | {"import_source.py"} <= set(approval["files"])
    for name, expected in approval["files"].items():
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE) and file_hash(path) == expected, (name, "approved input drift")
    assert file_hash(DATA.parent / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf") == approval["source_pdf_sha256"]
    return approval


def append_registry(role: str) -> bytes:
    current = TARGETS[role].read_bytes()
    existing = list(csv.DictReader(io.StringIO(current.decode())))
    proposed = list(csv.DictReader((HERE / INPUTS[role]).open(newline="")))
    assert len(proposed) == 1
    added = proposed[0]
    assert list(added) == list(existing[0]), (role, "registry header mismatch")
    assert all(r["ID"] != added["ID"] for r in existing), (role, "duplicate registry ID")
    if role == "languages":
        assert added["ID"] == "Kanikkaran" and added["Glottocode"] == "kani1275"
        assert not any(r["Glottocode"] == added["Glottocode"] for r in existing)
    else:
        assert added["ID"] == "nair2013-kerala-kani" and added["Language_ID"] == "Kanikkaran"
        assert not any(r["Tag"] == added["Tag"] or r["Source_Language_ID"] == added["Source_Language_ID"] for r in existing)
    assert not added["Latitude"] and not added["Longitude"] and not added["Quality"]
    extra = io.StringIO(newline="")
    csv.writer(extra).writerow([added[f] for f in added])
    return current + (b"" if current.endswith(b"\n") else b"\n") + extra.getvalue().encode()


def validate_shared_membership() -> None:
    for role in ("languages", "dialects"):
        expected = list(csv.DictReader((HERE / INPUTS[role]).open(newline="")))[0]
        actual = [r for r in csv.DictReader(TARGETS[role].open(newline="")) if r["ID"] == expected["ID"]]
        assert actual == [expected], (role, "installed registry contribution changed")
    expected = parse_string((HERE / INPUTS["bib"]).read_text(), "bibtex").entries
    actual = parse_string(TARGETS["bib"].read_text(), "bibtex").entries
    assert set(expected) <= set(actual)
    assert all(actual[k] == v for k, v in expected.items()), "installed bibliography contribution changed"


def prepare() -> tuple[dict, dict[Path, bytes]]:
    approval = validate_inputs()
    rows = list(csv.reader((HERE / INPUTS["csv"]).open(newline="")))
    assert len(rows) == approval["rows"] == 36
    assert all(len(r) == 15 and r[0] == "Kanikkaran" and r[2] and r[10] for r in rows)
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    assert all(not r[4] and not r[5] and not any(r[i] for i in (1, 8, 9, 11, 12, 13)) for r in rows)
    assert all(r[7] in {"nair2013kani[p. 32]", "nair2013kani[p. 33]"} for r in rows)
    assert all(unicodedata.is_normalized("NFC", r[2]) and not any(unicodedata.category(c) == "Co" or c == "\ufffd" for c in r[2]) for r in rows)
    audit = [json.loads(line) for line in (HERE / INPUTS["audit"]).read_text().splitlines()]
    assert len(audit) == len({u["source_unit_key"] for u in audit}) == approval["audit_units"] == 42
    assert sum(len(u["witnesses"]) for u in audit) == approval["witness_slots"] == 66
    assert {k for u in audit for k in u["entry_keys"]} == keys
    assert all(u["accepted_for_source_stage"] is True for u in audit)
    settings = yaml.safe_load((HERE / INPUTS["yaml"]).read_text())
    assert settings["file"] == STEM + ".csv"
    assert settings["defaults"]["transcription"] == {"convert": False, "preserve_hyphens": True}
    meta = SourceMeta([HERE / INPUTS["yaml"]])
    assert set(meta.sources) == {"nair2013kani", "somasekharannairkanikkarude"}
    assert meta.flag("nair2013kani", "identity", "dedupe_by_entry_key") is True
    assert settings["defaults"]["importer"]["commands"] == [["data/other/forms/raw_data/nair_kani_2013/import_source.py", "--install"]]
    old_bib = TARGETS["bib"].read_bytes()
    old = parse_string(old_bib.decode(), "bibtex").entries
    added = parse_string((HERE / INPUTS["bib"]).read_text(), "bibtex").entries
    assert set(added) == {"nair2013kani", "somasekharannairkanikkarude"}
    assert not set(added) & set(old), "citation keys already declared"
    bib = old_bib + (b"\n" if old_bib.endswith(b"\n") else b"\n\n") + (HERE / INPUTS["bib"]).read_bytes()
    merged = parse_string(bib.decode(), "bibtex").entries
    assert set(merged) - set(old) == set(added)
    assert all(merged[k] == v for k, v in old.items()), "unrelated bibliography changed"
    planned = {TARGETS[r]: (HERE / INPUTS[r]).read_bytes() for r in ("csv", "yaml", "audit")}
    planned.update({TARGETS["languages"]: append_registry("languages"), TARGETS["dialects"]: append_registry("dialects"), TARGETS["bib"]: bib})
    assert set(approval["canonical_baselines"]) == {str(p.relative_to(DATA)) for p in planned}
    assert all(file_hash(p) == approval["canonical_baselines"][str(p.relative_to(DATA))] for p in planned), "canonical baseline changed"
    assert file_hash(IDENTITY) == approval["durable_identity_sha256"]
    return approval, planned


def main(install: bool = False) -> dict:
    approval = validate_inputs()
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        assert file_hash(APPROVAL) == receipt["approval_sha256"]
        assert set(receipt["installed_sha256"]) == set(approval["canonical_baselines"])
        for role in ("csv", "yaml", "audit"):
            assert file_hash(TARGETS[role]) == receipt["installed_sha256"][str(TARGETS[role].relative_to(DATA))]
        validate_shared_membership()
        result = {"status": "already installed; own assets byte-exact and shared contributions intact", "rows": 36, "canonical_mutations": False}
        print(json.dumps(result)); return result
    if JOURNAL.exists():
        journal = json.loads(JOURNAL.read_text())
        assert file_hash(APPROVAL) == journal["approval_sha256"]
        assert set(journal["targets"]) == set(approval["canonical_baselines"])
        planned = {}
        for name, entry in journal["targets"].items():
            path = (DATA / name).resolve(); stage = (HERE / entry["staged_file"]).resolve()
            assert path.is_relative_to(DATA) and stage.is_relative_to(HERE / "installation-stage")
            assert file_hash(stage) == entry["output_sha256"]
            assert file_hash(path) in {approval["canonical_baselines"][name], entry["output_sha256"]}, (name, "partial transaction drift")
            planned[path] = stage.read_bytes()
        assert file_hash(IDENTITY) == approval["durable_identity_sha256"]
    else:
        approval, planned = prepare()
    result = {"status": "validated six-target Kani source-stage plan", "rows": 36, "audit_units": 42,
              "witness_slots": 66, "installed_sha256": {str(p.relative_to(DATA)): digest(b) for p, b in planned.items()},
              "approval_sha256": file_hash(APPROVAL), "durable_identity_sha256": file_hash(IDENTITY),
              "deferred": approval["deferred"]}
    if install:
        if not JOURNAL.exists():
            stage_dir = HERE / "installation-stage"; stage_dir.mkdir(exist_ok=True)
            targets = {}
            for n, (path, content) in enumerate(planned.items()):
                stage = stage_dir / f"{n:02d}.asset"
                assert not stage.exists() or file_hash(stage) == digest(content)
                stage.write_bytes(content)
                targets[str(path.relative_to(DATA))] = {"staged_file": str(stage.relative_to(HERE)), "output_sha256": digest(content)}
            journal = {"approval_sha256": file_hash(APPROVAL), "targets": targets}
            temp = JOURNAL.with_suffix(".tmp"); temp.write_text(json.dumps(journal, indent=2) + "\n"); temp.replace(JOURNAL)
        for path, content in planned.items():
            if file_hash(path) == digest(content):
                continue
            assert file_hash(path) == approval["canonical_baselines"][str(path.relative_to(DATA))]
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".kani-install-tmp")
            assert not temp.exists() or file_hash(temp) == digest(content)
            temp.write_bytes(content); temp.replace(path)
        assert all(file_hash(p) == digest(b) for p, b in planned.items()), "post-write target drift"
        assert file_hash(IDENTITY) == approval["durable_identity_sha256"]
        validate_shared_membership()
        result["status"] = "source-stage installed; full compiled gates deferred"
        temp = RECEIPT.with_suffix(".tmp"); temp.write_text(json.dumps(result, indent=2) + "\n"); temp.replace(RECEIPT)
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--install", action="store_true")
    main(parser.parse_args().install)
