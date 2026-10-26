"""Guarded full Irula chapter installation; never builds CLDF or DB.

Read-only by default. --install requires independently approved exact source
inputs; existing unexpected output or bibliography keys are never overwritten.
The existing Irula language registration is checked, not rewritten.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "tags.py").exists())
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta

PDF = DATA.parent / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
PDF_SHA = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
INPUTS = ["physical-units-first-reading.jsonl", "graph-decisions-proposal.json",
          "metadata-revisions-r2.json", "scope-revisions-r2.json", "same-cell-slash-graph-census.json",
          "proposed.csv", "proposal-audit.jsonl", "20260927-nair-irula.yaml",
          "proposed-reference.bib", "lowline-vector-census.json", "symbol-inventory.json", "metadata-reference-check.json"]
TARGETS = {"csv": DATA / "data/other/forms/20260927-nair-irula.csv",
           "yaml": DATA / "data/other/forms/20260927-nair-irula.yaml",
           "bib": DATA / "cldf/sources.bib", "audit": HERE / "installed-audit.jsonl"}
APPROVAL = HERE / "source-stage-approval.json"
RECEIPT = HERE / "installation-receipt.json"
IDENTITY = DATA / "data/form-identities.csv"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def current(path):
    return sha(path.read_bytes()) if path.exists() else None


def validate_approval():
    assert APPROVAL.exists(), "Independent exact source-stage approval is required"
    a = json.loads(APPROVAL.read_text())
    assert a["status"] == "approved" and a["material_errors"] == 0
    assert a["installer_sha256"] == current(Path(__file__)), "approved installer drift"
    assert a["source_pdf_sha256"] == PDF_SHA and current(PDF) == PDF_SHA
    assert set(a["input_sha256"]) == set(INPUTS)
    for name in INPUTS:
        assert current(HERE / name) == a["input_sha256"][name], (name, "reviewed input drift")
    return a


def prepare(*, require_approval):
    assert current(PDF) == PDF_SHA
    approval = validate_approval() if require_approval or APPROVAL.exists() else None
    rows = list(csv.reader((HERE / "proposed.csv").open()))
    assert len(rows) == 256 and len({r[10] for r in rows}) == 256
    assert all(len(r) == 15 and r[0] == "Irula" and r[7].startswith("nair2013irula[") for r in rows)
    keys = {r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows) == 12
    assert all(not r[11] or r[11] in keys for r in rows)
    assert sum(bool(r[13]) for r in rows) == 4
    assert all(not r[13] or all(k in keys for k in r[13].split("|")) for r in rows)
    audit = [json.loads(s) for s in (HERE / "proposal-audit.jsonl").read_text().splitlines()]
    assert len(audit) == 261
    assert {k for a in audit for k in a["proposed_entry_keys"]} == keys
    assert sum(a["disposition"] == "audit-only" for a in audit) == 16
    accepted = [{**a, "accepted_for_source_stage": True,
                 "acceptance_semantics": "The audit decision is accepted; audit-only units do not become lexical rows"} for a in audit]
    audit_bytes = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted).encode()
    config = HERE / "20260927-nair-irula.yaml"
    assert set(SourceMeta([config]).sources) == {"nair2013irula", "zvelebil1982irulapart3", "perialwar1978irula"}
    language = next(r for r in csv.DictReader((DATA / "cldf/languages.csv").open()) if r["ID"] == "Irula")
    assert language["Glottocode"] == "irul1243"

    old_bib = TARGETS["bib"].read_bytes()
    before = parse_string(old_bib.decode(), "bibtex").entries
    new_bib = (HERE / "proposed-reference.bib").read_bytes()
    merged = old_bib + b"\n" + new_bib
    after = parse_string(merged.decode(), "bibtex").entries
    assert set(after) - set(before) == {"nair2013irula", "zvelebil1982irulapart3", "perialwar1978irula"}
    assert all(after[k] == v for k, v in before.items())
    planned = {"csv": (HERE / "proposed.csv").read_bytes(), "yaml": config.read_bytes(),
               "bib": merged, "audit": audit_bytes}
    baselines = {role: current(path) for role, path in TARGETS.items()}
    assert all(baselines[r] is None for r in ("csv", "yaml", "audit")), "Unexpected existing source output"
    return planned, baselines, current(IDENTITY), {"bib": old_bib}, approval


def main(*, install):
    if RECEIPT.exists():
        validate_approval()
        r = json.loads(RECEIPT.read_text())
        assert all(current(TARGETS[k]) == r["outputs"][k] for k in ("csv", "yaml", "audit"))
        # Shared registries and durable IDs may legitimately advance after this
        # transaction; only this source's bibliography members must be unchanged.
        approved_bib = parse_string((HERE / "proposed-reference.bib").read_text(), "bibtex").entries
        live_bib = parse_string(TARGETS["bib"].read_text(), "bibtex").entries
        assert all(k in live_bib and live_bib[k] == v for k, v in approved_bib.items()), "source bibliography drift"
        print(json.dumps({"status": "already installed; exact byte no-op", "rows": 256}))
        return
    planned, baselines, identity, rollback, approval = prepare(require_approval=install)
    print(json.dumps({"status": "validated full-chapter proposal", "approved": approval is not None,
                      "install": install, "rows": 256, "audit_units": 261, "variant_edges": 12, "derived_edges": 4}, indent=2))
    if not install:
        return
    assert all(current(p) == baselines[k] for k, p in TARGETS.items())
    written = []
    try:
        for role, path in TARGETS.items():
            assert current(path) == baselines[role], (role, "concurrent target drift")
            temp = path.with_name(path.name + ".irula-install-tmp")
            assert not temp.exists()
            temp.write_bytes(planned[role])
            os.replace(temp, path)
            written.append(role)
        assert all(current(p) == sha(planned[k]) for k, p in TARGETS.items()), "post-write target drift"
        assert current(IDENTITY) == identity, "durable identity drift"
        r = {"status": "source-stage installed; full CLDF/browser gates deferred", "rows": 256,
             "audit_units": 261, "variant_edges": 12, "derived_edges": 4, "approval_sha256": current(APPROVAL),
             "source_pdf_sha256": PDF_SHA, "outputs": {k: sha(v) for k, v in planned.items()},
             "baselines": baselines, "durable_identity_sha256": identity}
        temp = RECEIPT.with_suffix(".tmp")
        assert not temp.exists()
        temp.write_text(json.dumps(r, indent=2) + "\n")
        os.replace(temp, RECEIPT)
    except BaseException:
        for role in reversed(written):
            path = TARGETS[role]
            if current(path) != sha(planned[role]):
                continue
            if baselines[role] is None:
                path.unlink()
            else:
                temp = path.with_name(path.name + ".irula-rollback-tmp")
                temp.write_bytes(rollback[role])
                os.replace(temp, path)
        raise
    print(json.dumps({"status": "installed", "rows": 256, "audit_units": 261, "variant_edges": 12, "derived_edges": 4}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", action="store_true")
    main(install=p.parse_args().install)
