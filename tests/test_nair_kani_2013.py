"""Kani source-stage preservation and guarded-install regressions; no full build."""
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unicodedata as ud
from pathlib import Path

import pytest
import yaml
from pybtex.database import parse_file

import make_cldf
import source_meta
from tags import GRAMMATICAL_TAGS

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA.parent
SOURCE = DATA / "data/other/forms/raw_data/nair_kani_2013"
REVIEW = SOURCE / "whole-recovery"
STEM = "20260927-nair-kani"
CSV = DATA / f"data/other/forms/{STEM}.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_installer():
    spec = importlib.util.spec_from_file_location("kani_installer_fixture", SOURCE / "import_source.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def installer_fixture():
    parent = ROOT / "tmp/kani-installer-tests"; parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        root = Path(temp); data = root / "data"; here = data / "data/other/forms/raw_data/nair_kani_2013"
        here.mkdir(parents=True); (data / "cldf").mkdir()
        approval = json.loads((SOURCE / "source-stage-authorization.json").read_text())
        for name in approval["files"]:
            target = here / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, target)
        for name in ("languages.csv", "dialects.csv", "sources.bib"):
            shutil.copyfile(DATA / "cldf" / name, data / "cldf" / name)
        # Fixtures may run after installation: remove only this source's new
        # membership from the fixture copies to reconstruct a clean baseline.
        for name, added_id in (("languages.csv", "Kanikkaran"), ("dialects.csv", "nair2013-kerala-kani")):
            path = data / "cldf" / name
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream); fields = reader.fieldnames; records = [r for r in reader if r["ID"] != added_id]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(records)
        bib = parse_file(str(data / "cldf/sources.bib"))
        for key in ("nair2013kani", "somasekharannairkanikkarude"):
            bib.entries.pop(key, None)
        (data / "cldf/sources.bib").write_text(bib.to_string("bibtex"))
        identity = data / "data/form-identities.csv"; identity.write_text("fixture identity must survive\n")
        pdf = root / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"; pdf.parent.mkdir(parents=True)
        pdf.symlink_to(ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf")
        module = load_installer(); module.HERE = here; module.DATA = data
        module.APPROVAL = here / "source-stage-authorization.json"
        module.RECEIPT = here / "installation-receipt.json"
        module.JOURNAL = here / "installation-transaction.json"
        module.IDENTITY = identity
        module.TARGETS = {"csv": data / f"data/other/forms/{STEM}.csv", "yaml": data / f"data/other/forms/{STEM}.yaml", "audit": here / "installed-audit.jsonl", "languages": data / "cldf/languages.csv", "dialects": data / "cldf/dialects.csv", "bib": data / "cldf/sources.bib"}
        approval["canonical_baselines"] = {str(p.relative_to(data)): sha(p) for p in module.TARGETS.values()}
        approval["durable_identity_sha256"] = sha(identity)
        module.APPROVAL.write_text(json.dumps(approval))
        yield module


def test_installer_clean_and_repeat_are_guarded(installer_fixture):
    module = installer_fixture; baselines = {k: sha(p) for k, p in module.TARGETS.items()}
    module.main(False)
    assert {k: sha(p) for k, p in module.TARGETS.items()} == baselines
    module.main(True)
    after = {k: sha(p) for k, p in module.TARGETS.items()}
    module.main(True)
    assert {k: sha(p) for k, p in module.TARGETS.items()} == after
    assert module.IDENTITY.read_text() == "fixture identity must survive\n"
    assert len(json.loads(module.JOURNAL.read_text())["targets"]) == 6


def test_installer_rejects_unapproved_input(installer_fixture):
    module = installer_fixture
    path = module.HERE / module.INPUTS["csv"]; path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match="approved input drift"):
        module.main(True)
    assert not module.TARGETS["csv"].exists()


def test_installer_resumes_partial_write(installer_fixture, monkeypatch):
    module = installer_fixture; original = Path.replace; failed = False
    def interrupted(path, target):
        nonlocal failed
        if Path(target) == module.TARGETS["languages"] and not failed:
            failed = True; raise OSError("simulated interruption")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        module.main(True)
    assert module.JOURNAL.exists() and not module.RECEIPT.exists()
    module.main(True)
    assert module.RECEIPT.exists()
    module.main(False)


def test_installer_detects_postwrite_drift(installer_fixture, monkeypatch):
    module = installer_fixture; original = Path.replace
    def tampered(path, target):
        result = original(path, target)
        if Path(target) == module.TARGETS["bib"]:
            module.TARGETS["csv"].write_bytes(b"unexpected concurrent edit\n")
        return result
    monkeypatch.setattr(Path, "replace", tampered)
    with pytest.raises(AssertionError, match="post-write target drift"):
        module.main(True)
    assert not module.RECEIPT.exists()
    with pytest.raises(AssertionError, match="partial transaction drift"):
        module.main(False)


def test_installed_counts_literals_and_dispositions():
    rows = list(csv.reader(CSV.open(newline="")))
    audit = [json.loads(line) for line in (SOURCE / "installed-audit.jsonl").read_text().splitlines()]
    assert len(rows) == len({r[10] for r in rows}) == 36
    assert CSV.read_bytes() == (REVIEW / "review-candidate.csv").read_bytes()
    assert len(audit) == len({r["source_unit_key"] for r in audit}) == 42
    assert sum(len(a["witnesses"]) for a in audit) == 66
    keys = {r[10] for r in rows}
    assert {k for a in audit for k in a["entry_keys"]} == keys
    assert sum(w["status"] == "excluded_control_or_documentary" for a in audit for w in a["witnesses"]) == 30
    assert all(a["accepted_for_source_stage"] for a in audit)
    assert next(r for r in rows if r[10] == "nair2013kani:p33:negative:0:w1")[2] == "aa"
    assert {"kooyiine", "naṭakkaan parraa", "meya peyyine", "uumakkiyaŋŋu"} <= {r[2] for r in rows}
    assert all(not any(r[i] for i in (1, 4, 5, 8, 9, 11, 12, 13)) for r in rows)
    assert all(ud.is_normalized("NFC", r[2]) and not any(ud.category(c) == "Co" or c == "\ufffd" for c in r[2]) for r in rows)


def test_installed_registry_reference_and_policy():
    meta = source_meta.SourceMeta(source_meta.yaml_files())
    settings = yaml.safe_load((CSV.with_suffix(".yaml")).read_text())
    assert settings["defaults"]["transcription"] == {"convert": False, "preserve_hyphens": True}
    assert meta.flag("nair2013kani", "identity", "dedupe_by_entry_key") is True
    assert meta.flag("nair2013kani", "forms", "split_alternates") is False
    assert meta.flag("nair2013kani", "reference", "ocr") is False
    assert meta.flag("nair2013kani", "reference", "etymology_provenance") == "none"
    languages = [r for r in csv.DictReader((DATA / "cldf/languages.csv").open()) if r["ID"] == "Kanikkaran"]
    assert len(languages) == 1 and languages[0]["Glottocode"] == "kani1275"
    assert languages[0]["Clade"] == "S. Dravidian I"
    assert not languages[0]["Latitude"] and not languages[0]["Longitude"] and not languages[0]["Quality"]
    dialects = [r for r in csv.DictReader((DATA / "cldf/dialects.csv").open()) if r["ID"] == "nair2013-kerala-kani"]
    assert len(dialects) == 1 and dialects[0]["Language_ID"] == "Kanikkaran"
    assert not dialects[0]["Latitude"] and not dialects[0]["Longitude"]
    rows = list(csv.reader(CSV.open()))
    assert all(r[0] == "Kanikkaran" and dialects[0]["Tag"] in r[14].split() for r in rows)
    assert all(t in GRAMMATICAL_TAGS or t == dialects[0]["Tag"] for r in rows for t in r[14].split())
    refs = parse_file(str(DATA / "cldf/sources.bib")).entries
    assert refs["nair2013kani"].fields["ocr"] == "No"
    assert refs["nair2013kani"].fields["pages"] == "32--33"
    assert "year" not in refs["somasekharannairkanikkarude"].fields


def test_actual_canonical_parser_nfc_nfd_and_initial_identity():
    rows = list(csv.reader(CSV.open(newline="")))
    parent = ROOT / "tmp/kani-parser-tests"; parent.mkdir(exist_ok=True)
    fields = ("lang", "param", "old_form", "gloss", "native", "ipa", "notes", "source", "cognateset", "etymology", "entry_key", "variant_of_key", "borrowed_from_key", "derivation_parent_keys", "tags")
    meta = source_meta.load()
    canonical_errors = io.StringIO()
    canonical, canonical_stats = make_cldf.parse_file(str(CSV), canonical_errors, name=STEM)
    assert canonical_errors.getvalue() == "" and len(canonical) == 36
    assert [[getattr(r, f) for f in fields] for r in canonical] == rows
    assert canonical_stats == {"converted": 0, "for_conversion": 0}
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        fixture = Path(temporary) / "data/other/forms" / (STEM + ".csv")
        fixture.parent.mkdir(parents=True)
        for normalization in ("NFC", "NFD"):
            expected = [[ud.normalize(normalization, v) if i == 2 else v for i, v in enumerate(r)] for r in rows]
            with fixture.open("w", newline="") as out: csv.writer(out).writerows(expected)
            errors = io.StringIO(); parsed, stats = make_cldf.parse_file(str(fixture), errors, name=STEM)
            assert errors.getvalue() == "" and len(parsed) == 36
            assert stats == {"converted": 0, "for_conversion": 0}
            assert [[getattr(r, f) for f in fields] for r in parsed] == expected
            assert all(ud.normalize("NFC", r.form) == ud.normalize("NFC", raw[2]) for r, raw in zip(parsed, expected))
            assert len({make_cldf.source_entry_dedupe_key(r, meta) for r in parsed}) == 36


def test_installed_receipt_and_independent_review_continuity():
    receipt = json.loads((SOURCE / "installation-receipt.json").read_text())
    approval = json.loads((SOURCE / "source-stage-authorization.json").read_text())
    assert receipt["approval_sha256"] == sha(SOURCE / "source-stage-authorization.json")
    assert set(receipt["installed_sha256"]) == set(approval["canonical_baselines"])
    assert len(receipt["installed_sha256"]) == 6
    assert receipt["durable_identity_sha256"] == approval["durable_identity_sha256"]
    continuity = json.loads((REVIEW / "revision2-continuity.json").read_text())
    assert continuity["changed_entry_keys"] == ["nair2013kani:p33:negative:0:w1"]
    addendum = json.loads((REVIEW / "independent-sample/independent-original-review-pass2-addendum.json").read_text())
    assert addendum["control_result"] == "pass" and addendum["unchanged_selected_source_units"] == 20
