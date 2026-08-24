import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/merriam_dravidian_db.py"
SPEC = importlib.util.spec_from_file_location("merriam_dravidian_db_importer", SCRIPT)
source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = source
SPEC.loader.exec_module(source)


def installed_rows():
    with source.OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def audit_rows():
    with gzip.open(source.AUDIT, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exact_counts_and_complete_audit():
    installed = installed_rows()
    audit = audit_rows()
    assert len(audit) == 6697
    assert len(installed) == 6672
    assert {len(row) for row in installed} == {15}
    statuses = {}
    for row in audit:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    assert statuses == {"ingested": 6666, "ambiguous": 17, "unresolved": 8, "unlinked": 6}


def test_levels_ids_sources_and_transcription_are_preserved():
    installed = installed_rows()
    by_key = {row[10]: row for row in installed}
    assert len(by_key) == len(installed)

    pd = by_key["merriam2026dravidiandb:pdr:1"]
    assert pd[:4] == ["PDr", "d1", "a-", "that; that woman or thing; look there!; that place, there; there; so much, all, whole; that time, then, afterwards; that time, then; thus; that man; those things; adj.that; that woman; those persons; that day; thus, in that way; all; then, at that time; there, in that place; there, thence; that man, those men, that woman, those women, that thing, those things; those men; that (remote); those women or things; there!; then; therefore; this"]
    assert pd[7].startswith("merriam2026dravidiandb[record 1, DEDR 1];starostin2006dravidian")

    km = by_key["merriam2026dravidiandb:pdr:18"]
    assert km[:4] == ["PKMDr", "d17", "āq-", "to know"]

    unlinked = by_key["merriam2026dravidiandb:pdr:5904"]
    assert unlinked[:4] == ["PSD2", "", "nā", "obl of 1sg"]

    assert all("�" not in "".join(row) for row in installed)


def test_ambiguous_and_invalid_dedr_targets_are_not_installed():
    installed_keys = {row[10] for row in installed_rows()}
    audit = {row["id_pdr"]: row for row in audit_rows()}
    assert audit["612"]["status"] == "ambiguous"
    assert audit["612"]["parameter_id"] == "d583"
    assert "merriam2026dravidiandb:pdr:612" not in installed_keys
    assert audit["2375"]["status"] == "unresolved"
    assert audit["2375"]["parameter_id"] == "d2187"
    assert "merriam2026dravidiandb:pdr:2375" not in installed_keys


def test_manifest_and_compiled_rows():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == source.SOURCE_SHA256
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["installed_rows"] == 6672

    compiled_path = ROOT / "cldf/forms.csv"
    if not compiled_path.exists():
        return
    with compiled_path.open(encoding="utf-8", newline="") as handle:
        compiled = list(csv.DictReader(handle))
    source_rows = [
        row for row in compiled
        if "merriam2026dravidiandb[" in row["Source"] and row["Status"] != "entry"
    ]
    # The focused source tests may run before the full CLDF rebuild.
    if not source_rows:
        return
    assert len(source_rows) == 6672
    assert {row["Language_ID"] for row in source_rows} == set(source.CLASSIFICATION_LANGUAGE.values())
    assert all(row["Form"].startswith("*") for row in source_rows)
    record_one = next(row for row in source_rows if "record 1," in row["Source"])
    assert record_one["Form"] == "*a-"
    assert record_one["Original"] == "a-"

    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as handle:
        keys = [
            row for row in csv.DictReader(handle)
            if row["Source_Key"].startswith("merriam2026dravidiandb:pdr:")
        ]
    assert len(keys) == 6672
    assert len({row["Source_Key"] for row in keys}) == 6672


def test_seeded_review_sample():
    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert len({row["id_pdr"] for row in rows}) == 20
    assert {row["review_result"] for row in rows} == {"pass"}
    assert {row["status"] for row in rows} == {"ingested"}
    assert len({row["language_id"] for row in rows}) >= 4
