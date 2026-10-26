"""Guarded Malamuthan full-chapter source-stage transaction; never builds CLDF/DB.

Read-only by default. --install requires a separate independently reviewed
source-stage-approval.json pinned to exact local input bytes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "tags.py").exists())
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta

WORKSPACE = DATA.parent
PDF = WORKSPACE / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
PDF_SHA = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
INPUTS = ["physical-units-first-reading.jsonl", "graph-decisions-proposal.json",
          "proposed.csv", "proposal-audit.jsonl", "20260927-nair-malamuthan.yaml",
          "proposed-language.csv", "proposed-reference.bib"]
TARGETS = {
    "csv": DATA / "data/other/forms/20260927-nair-malamuthan.csv",
    "yaml": DATA / "data/other/forms/20260927-nair-malamuthan.yaml",
    "languages": DATA / "cldf/languages.csv",
    "bib": DATA / "cldf/sources.bib",
    "audit": HERE / "installed-audit.jsonl",
}
APPROVAL = HERE / "source-stage-approval.json"
RECEIPT = HERE / "installation-receipt.json"
IDENTITY = DATA / "data/form-identities.csv"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def current(path):
    return sha(path.read_bytes()) if path.exists() else None


def validate_approval():
    assert APPROVAL.exists(), "Independent source-stage approval is required"
    approval = json.loads(APPROVAL.read_text())
    assert approval["status"] == "approved" and approval["material_errors"] == 0
    assert approval["source_pdf_sha256"] == PDF_SHA
    assert current(PDF) == PDF_SHA
    assert set(approval["input_sha256"]) == set(INPUTS)
    for name in INPUTS:
        assert current(HERE / name) == approval["input_sha256"][name], (name, "reviewed input drift")
    return approval


def prepare(*, require_approval):
    assert current(PDF) == PDF_SHA
    approval = validate_approval() if require_approval or APPROVAL.exists() else None
    rows = list(csv.reader((HERE / "proposed.csv").open()))
    assert len(rows) == 67 and len({r[10] for r in rows}) == 67
    assert all(len(r) == 15 and r[0] == "Malamuthan" and r[7].startswith("nair2013malamuthan[") for r in rows)
    keys = {r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows) == 2
    assert all(not r[11] or r[11] in keys for r in rows)
    audit = [json.loads(s) for s in (HERE / "proposal-audit.jsonl").read_text().splitlines()]
    assert len(audit) == 79
    assert {a["proposed_entry_key"] for a in audit if a["proposed_entry_key"]} == keys
    accepted = [{**a, "accepted_for_source_stage": True} for a in audit]
    audit_bytes = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted).encode()

    config = HERE / "20260927-nair-malamuthan.yaml"
    assert "nair2013malamuthan" in SourceMeta([config]).sources
    old_lang = TARGETS["languages"].read_bytes()
    lang_rows = list(csv.DictReader(io.StringIO(old_lang.decode())))
    assert lang_rows and not any(r["ID"] == "Malamuthan" for r in lang_rows)
    proposed_lang = list(csv.reader((HERE / "proposed-language.csv").open()))
    assert len(proposed_lang) == 2 and proposed_lang[0] == list(lang_rows[0])
    assert proposed_lang[1][0] == "Malamuthan" and len(proposed_lang[1]) == len(proposed_lang[0])
    extra = io.StringIO(newline="")
    csv.writer(extra).writerow(proposed_lang[1])
    lang_bytes = old_lang + extra.getvalue().encode()

    old_bib = TARGETS["bib"].read_bytes()
    prior = parse_string(old_bib.decode(), "bibtex").entries
    proposed_bib = (HERE / "proposed-reference.bib").read_bytes()
    merged = old_bib + b"\n" + proposed_bib
    after = parse_string(merged.decode(), "bibtex").entries
    assert set(after) - set(prior) == {"nair2013malamuthan", "somasekharannair1981malamuthan"}
    assert all(after[k] == v for k, v in prior.items())

    planned = {"csv": (HERE / "proposed.csv").read_bytes(), "yaml": config.read_bytes(),
               "languages": lang_bytes, "bib": merged, "audit": audit_bytes}
    baseline_hashes = {role: current(path) for role, path in TARGETS.items()}
    assert baseline_hashes["csv"] is None and baseline_hashes["yaml"] is None
    assert baseline_hashes["audit"] is None
    rollback_bytes = {"languages": old_lang, "bib": old_bib}
    return planned, baseline_hashes, current(IDENTITY), rollback_bytes, approval


def main(*, install):
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        assert all(current(TARGETS[role]) == expected for role, expected in receipt["outputs"].items())
        assert current(IDENTITY) == receipt["durable_identity_sha256"]
        print(json.dumps({"status": "already installed; exact byte no-op", "rows": 67}))
        return
    planned, baselines, identity, rollback_bytes, approval = prepare(require_approval=install)
    print(json.dumps({"status": "validated complete-chapter proposal", "install": install,
                      "approved": approval is not None, "rows": 67, "audit_units": 79,
                      "variant_edges": 2, "baselines": baselines}, indent=2))
    if not install:
        return
    assert all(current(path) == baselines[role] for role, path in TARGETS.items())
    written = []
    try:
        for role, path in TARGETS.items():
            assert current(path) == baselines[role], (role, "concurrent target drift")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".malamuthan-install-tmp")
            assert not temp.exists()
            temp.write_bytes(planned[role])
            os.replace(temp, path)
            written.append(role)
        assert all(current(path) == sha(planned[role]) for role, path in TARGETS.items()), "post-write target drift"
        assert current(IDENTITY) == identity, "durable identity drift"
        receipt = {"status": "source-stage installed; full CLDF/browser gates deferred",
                   "rows": 67, "audit_units": 79, "variant_edges": 2,
                   "approval_sha256": current(APPROVAL), "source_pdf_sha256": PDF_SHA,
                   "outputs": {role: sha(data) for role, data in planned.items()},
                   "baselines": baselines, "durable_identity_sha256": identity}
        temp = RECEIPT.with_suffix(".tmp")
        assert not temp.exists()
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temp, RECEIPT)
    except BaseException:
        for role in reversed(written):
            path = TARGETS[role]
            if current(path) != sha(planned[role]):
                continue  # Never overwrite a concurrent edit.
            if baselines[role] is None:
                path.unlink()
            else:
                original = rollback_bytes.get(role)
                assert original is not None, "Missing rollback baseline"
                temp = path.with_name(path.name + ".malamuthan-rollback-tmp")
                temp.write_bytes(original)
                os.replace(temp, path)
        raise
    print(json.dumps({"status": "installed", "rows": 67, "audit_units": 79, "variant_edges": 2}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    main(install=parser.parse_args().install)
