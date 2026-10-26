"""Guarded Kurichiya whole-chapter source-stage installation; no DB build.

Default invocation only validates and previews. --install writes the canonical CSV,
YAML, language/Bib append, accepted source audit and a receipt atomically per file.
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

ROOT = DATA.parent
PDF = ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED = {
    "proposed.csv": "ef3956895f096896757dc2209cf872718a838fa9ec8ee2caf3246bfce5822210",
    "proposal-audit.jsonl": "48136fbe7e1ffd5df324e6642a3d44cccd1fd50a781e9423dbff12064cf84d19",
    "first-reading/unicode-preview.jsonl": "a0bf4c4529128f72762162619db42bbf0b1fc0c27bee9c6c37d9321e684de458",
    "graph-review-proposal.json": "19cf7f4182824a5090ddd13a517c464119db1b4bc41f117bdc876a4b73b2b90a",
    "20260927-nair-kurichiya.yaml": "52c1f1b204b33436a97ad3f5ec9b184c6167cf2e7836ea3ded39258a0debdd6b",
    "proposed-reference.bib": "25426b9437d25a71027e992ecb9c2ceec228853f1667d1198cdb582a13dd26c8",
    "proposed-language.csv": "19aed50ecd0f6a091f394ff77486ef3fb62e8df272d2d87f128d70ebad75f8ed",
}
PDF_SHA = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
REVIEW = HERE / "independent-review/postgraph-fresh20-20260927404.json"
REVIEW_SHA = "d342ac41e258338380169573307e56f1a0fc461dbe4dc738fbb455a1c50a9550"
TARGETS = {
    "csv": DATA / "data/other/forms/20260927-nair-kurichiya.csv",
    "yaml": DATA / "data/other/forms/20260927-nair-kurichiya.yaml",
    "languages": DATA / "cldf/languages.csv",
    "bib": DATA / "cldf/sources.bib",
    "audit": HERE / "installed-audit.jsonl",
}
RECEIPT = HERE / "installation-receipt.json"
IDENTITY = DATA / "data/form-identities.csv"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def current(path):
    return digest(path.read_bytes()) if path.exists() else None


def prepare():
    for path, expected in EXPECTED.items():
        assert current(HERE / path) == expected, (path, "frozen source changed")
    assert current(PDF) == PDF_SHA
    assert current(REVIEW) == REVIEW_SHA
    review = json.loads(REVIEW.read_text())
    assert review["material_errors"] == 0 and len(review["sampled"]) == 20
    assert review["csv_sha256"] == EXPECTED["proposed.csv"]
    assert review["audit_sha256"] == EXPECTED["proposal-audit.jsonl"]
    assert review["graph_review"]["edge_count"] == 8
    assert review["graph_review"]["all_endpoints_in_frozen_csv"]
    rows = list(csv.reader((HERE / "proposed.csv").open()))
    assert len(rows) == 118 and len({r[10] for r in rows}) == 118
    assert sum(r[0] == "Kurichiya" for r in rows) == 71
    assert sum(r[0] == "Malayalam" for r in rows) == 47
    assert all(len(r) == 15 and r[7].startswith("nair2013kurichiya[") for r in rows)
    keys = {r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows) == 8
    assert all(not r[11] or r[11] in keys for r in rows)
    audit = [json.loads(line) for line in (HERE / "proposal-audit.jsonl").read_text().splitlines()]
    assert len(audit) == 117 and len({a["witness_key"] for a in audit}) == 117
    assert {k for a in audit for k in a.get("proposed_entry_keys", [])} == keys
    assert sum(a["disposition"] == "audit-only repeated prose" for a in audit) == 1
    accepted = [{**a, "accepted_for_source_stage": True} for a in audit]
    audit_bytes = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted).encode()
    profile = HERE / "20260927-nair-kurichiya.yaml"
    assert "nair2013kurichiya" in SourceMeta([profile]).sources

    old_languages = TARGETS["languages"].read_bytes()
    language_rows = list(csv.DictReader(io.StringIO(old_languages.decode())))
    assert not any(r["ID"] == "Kurichiya" for r in language_rows)
    proposed_language = list(csv.reader((HERE / "proposed-language.csv").open()))
    assert len(proposed_language) == 2 and proposed_language[0] == list(language_rows[0].keys())
    assert proposed_language[1][0] == "Kurichiya" and proposed_language[1][2] == "kuri1256"
    extra = io.StringIO(newline="")
    csv.writer(extra).writerow(proposed_language[1])
    assert old_languages.endswith(b"\n")
    languages = old_languages + extra.getvalue().encode()

    old_bib = TARGETS["bib"].read_bytes()
    old_refs = parse_string(old_bib.decode(), "bibtex").entries
    addition = (HERE / "proposed-reference.bib").read_bytes()
    bib = old_bib + b"\n" + addition
    new_refs = parse_string(bib.decode(), "bibtex").entries
    assert set(new_refs) - set(old_refs) == {"nair2013kurichiya", "vayaleri1992kurichiya"}
    assert all(new_refs[k] == v for k, v in old_refs.items())
    planned = {"csv": (HERE / "proposed.csv").read_bytes(), "yaml": profile.read_bytes(),
               "languages": languages, "bib": bib, "audit": audit_bytes}
    baseline = {role: current(path) for role, path in TARGETS.items()}
    assert baseline["csv"] is baseline["yaml"] is baseline["audit"] is None
    return planned, baseline, current(IDENTITY), {"languages": old_languages, "bib": old_bib}


def main(install=False):
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        assert all(current(TARGETS[role]) == expected for role, expected in receipt["outputs"].items())
        assert current(IDENTITY) == receipt["durable_identity_sha256"]
        print(json.dumps({"status": "already installed, byte-exact no-op", "rows": 118}))
        return
    planned, baseline, identity, originals = prepare()
    print(json.dumps({"status": "validated complete-chapter source-stage proposal", "install": install,
                      "rows": 118, "audit_units": 117, "variant_edges": 8, "baselines": baseline}, indent=2))
    if not install:
        return
    assert all(current(path) == baseline[role] for role, path in TARGETS.items())
    written = []
    try:
        for role, path in TARGETS.items():
            assert current(path) == baseline[role], (role, "concurrent target change")
            temp = path.with_name(path.name + ".kurichiya-install-tmp")
            assert not temp.exists()
            temp.write_bytes(planned[role])
            os.replace(temp, path)
            written.append(role)
        assert all(current(path) == digest(planned[role]) for role, path in TARGETS.items())
        assert current(IDENTITY) == identity
        receipt = {"status": "source-stage installed; full CLDF/browser gates deferred", "rows": 118,
                   "audit_units": 117, "variant_edges": 8, "frozen_inputs": EXPECTED,
                   "source_pdf_sha256": PDF_SHA, "independent_review_sha256": REVIEW_SHA,
                   "outputs": {role: digest(planned[role]) for role in TARGETS},
                   "baselines": baseline, "durable_identity_sha256": identity}
        temp = RECEIPT.with_suffix(".tmp")
        assert not temp.exists()
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temp, RECEIPT)
    except BaseException:
        for role in reversed(written):
            path = TARGETS[role]
            if current(path) != digest(planned[role]):
                continue
            if baseline[role] is None:
                path.unlink()
            else:
                original = originals.get(role)
                assert original is not None
                temp = path.with_name(path.name + ".kurichiya-rollback-tmp")
                temp.write_bytes(original)
                os.replace(temp, path)
        raise
    print(json.dumps({"status": "installed", "rows": 118, "audit_units": 117, "variant_edges": 8}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    main(parser.parse_args().install)
