import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/pinnow_juang_1960.py"
SPEC = importlib.util.spec_from_file_location("pinnow_juang_1960_importer", SCRIPT)
source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = source
SPEC.loader.exec_module(source)


def installed_rows():
    with source.OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def audit_rows():
    with source.AUDIT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_audit_rows():
    first = {}
    for row in audit_rows():
        first.setdefault(row["Raw_ID"], row)
    return first


def test_complete_source_variant_and_duplicate_accounting():
    installed, audit, first = installed_rows(), audit_rows(), first_audit_rows()
    assert len(first) == 1658
    assert len(audit) == 1824
    assert len(installed) == 1818
    assert {len(row) for row in installed} == {15}
    assert len({row[10] for row in installed}) == 1818
    assert Counter(row["Status"] for row in audit) == {"ingested": 1818, "excluded": 6}
    excluded = [row for row in audit if row["Status"] == "excluded"]
    assert {row["Raw_ID"] for row in excluded} == {
        "pinnow1960beitraege:C:i508", "pinnow1960beitraege:C:i523",
        "pinnow1960beitraege:C:i679", "pinnow1960beitraege:C:i784",
        "pinnow1960beitraege:C:i1437", "pinnow1960beitraege:C:i1579",
    }
    assert {row["Reason"] for row in excluded} == {
        "exact repeated alternant inside one source record"
    }
    assert {row[0] for row in installed} == {"ju"}
    assert all(row[2] == row[5] for row in installed)


def test_proto_munda_links_replace_only_secure_legacy_rows():
    first = first_audit_rows()
    linked = [row for row in first.values() if row["Parameter_ID"]]
    assert len(linked) == 66
    assert {row["Parameter_ID"] for row in linked} == set(source.PROTO_LINKS.values())
    assert sum(bool(row[1]) for row in installed_rows()) == 72
    assert all(row["Rau_Citation"].startswith("PJDW[") for row in linked)
    assert all("Rau 2019 assigns" in row["Final_Etymology"] for row in linked)

    tongue = [row for row in audit_rows() if row["Raw_ID"] == "pinnow1960beitraege:C:i477"]
    assert [row["Final_Form"] for row in tongue] == ["elaŋ", "ɛlaŋ"]
    assert {row["Parameter_ID"] for row in tongue} == {"m3"}
    assert tongue[1]["Variant_Of_Key"] == tongue[0]["Entry_Key"]

    with source.LEGACY_FORMS.open(encoding="utf-8", newline="") as handle:
        legacy = [row for row in csv.reader(handle) if len(row) >= 8 and row[7].startswith("PJDW[")]
    assert len(legacy) == 7
    assert {row[1] for row in legacy} == source.LEGACY_ONLY_PARAMETERS


def test_notes_gloss_queries_and_transcription_uncertainty():
    first = first_audit_rows()
    assert Counter(row["Note_Class"] for row in first.values()) == {
        "comparative": 1400, "none": 248, "comment": 10,
    }
    assert sum(not row["Final_Gloss"] for row in first.values()) == 185

    comparison = first["pinnow1960beitraege:C:i1335"]
    assert comparison["Final_Gloss"] == "to milk"
    assert comparison["Final_Etymology"].startswith("#Kh./rɔ'j/")
    assert "Proto-Munda m94" in comparison["Final_Etymology"]
    assert not comparison["Final_Notes"]

    queried = first["pinnow1960beitraege:C:i603"]
    assert queried["Raw_Gloss"] == "?"
    assert not queried["Final_Gloss"]
    assert queried["Final_Notes"] == "Source gloss: ?"
    assert queried["Tags"] == "uncertain"

    cleaned = [row for row in audit_rows() if row["Raw_ID"] == "pinnow1960beitraege:C:i679"]
    assert [row["Final_Form"] for row in cleaned] == ["iɲam", "iɲam"]
    assert [row["Status"] for row in cleaned] == ["ingested", "excluded"]
    assert cleaned[1]["Tags"] == "uncertain"

    elwin = first["pinnow1960beitraege:C:i32"]
    assert elwin["Raw_Form"] == "agmutri(E54,575)"
    assert elwin["Final_Form"] == "agmutri"
    assert elwin["Final_Notes"] == "Source form marker: (E54,575)"


def test_manifest_profile_sample_and_offline_rebuild():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 1658
    assert manifest["source_variant_rows"] == 1824
    assert manifest["installed_rows"] == 1818
    assert manifest["excluded_rows"] == 6
    assert manifest["proto_munda_linked_source_records"] == 66
    assert manifest["proto_munda_linked_installed_rows"] == 72
    assert manifest["legacy_pjdw_rows_replaced"] == 66
    assert set(manifest["legacy_pjdw_rows_retained"]) == source.LEGACY_ONLY_PARAMETERS
    assert manifest["policy"]["extraction"].endswith("no OCR")

    profile = source.PROFILE.read_text(encoding="utf-8")
    assert profile.startswith("Grapheme\tIPA\n \t#\n")
    assert "�" not in profile
    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == len({row["Raw_ID"] for row in sample}) == 20
    assert sum(row["Link_Status"] == "linked" for row in sample) >= 6
    assert sum(row["Note_Class"] == "comparative" for row in sample) >= 4
    assert {row["Review_Result"] for row in sample} == {"pass"}
    assert {row["Material_Error"] for row in sample} == {""}

    rebuilt, audit = source.transform(source.offline_records())
    assert rebuilt == installed_rows()
    assert audit == audit_rows()


def test_compiled_rows_survive_when_current_cldf_is_built():
    path = ROOT / "cldf/forms.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if "Entry_Key" in (all_rows[0] if all_rows else {}):
        rows = [r for r in all_rows if r.get("Entry_Key", "").startswith(source.SOURCE_KEY + ":")]
        linked = sum(bool(row["Parameter_ID"]) for row in rows)
    else:
        compiled = {row["ID"]: row for row in all_rows}
        with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as handle:
            key_rows = [
                row for row in csv.DictReader(handle)
                if row["Source_Key"].startswith(source.SOURCE_KEY + ":")
            ]
        ids = {row["Legacy_ID"] for row in key_rows}
        rows = [compiled[form_id] for form_id in ids]
        linked = sum(bool(__import__("re").match(r"m\d+-", form_id)) for form_id in ids)
    if not rows:
        return
    assert len(rows) == 1818
    assert {row["Language_ID"] for row in rows} == {"ju"}
    assert linked == 72


SOURCE_KEY = source.SOURCE_KEY
