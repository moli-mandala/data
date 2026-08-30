import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/bahl_korwa_1962.py"
SPEC = importlib.util.spec_from_file_location("bahl_korwa_1962_importer", SCRIPT)
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


def test_complete_source_variant_and_exclusion_accounting():
    installed = installed_rows()
    audit = audit_rows()
    first = first_audit_rows()
    assert len(first) == 1792
    assert len(audit) == 1831
    assert len(installed) == 1830
    assert {len(row) for row in installed} == {15}
    assert len({row[10] for row in installed}) == 1830
    assert Counter(row["Status"] for row in audit) == {
        "ingested": 1830, "excluded": 1,
    }
    excluded = next(row for row in audit if row["Status"] == "excluded")
    assert excluded["Raw_ID"] == "bahl1962korwa:C:c1.p133.r16.i1805"
    assert not excluded["Raw_Form"] and not excluded["Raw_Gloss"]
    assert excluded["Reason"] == "empty source record"
    assert {row[0] for row in installed} == {"kw"}
    assert all(row[2] == row[5] for row in installed)


def test_proto_munda_links_replace_only_secure_legacy_rows():
    first = first_audit_rows()
    linked = [row for row in first.values() if row["Parameter_ID"]]
    assert len(linked) == 57
    assert {row["Parameter_ID"] for row in linked} == set(source.PROTO_LINKS.values())
    assert sum(bool(row[1]) for row in installed_rows()) == 58
    assert {row["Alignment_Method"] for row in linked} == {
        "unique normalized source form plus compatible meaning"
    }
    assert all(row["Rau_Citation"].startswith("BAHL[") for row in linked)
    assert all("Rau 2019 assigns" in row["Final_Etymology"] for row in linked)

    death = [
        row for row in audit_rows()
        if row["Raw_ID"] == "bahl1962korwa:C:c1.p46.r11.i674"
    ]
    assert [row["Final_Form"] for row in death] == ["goej", "goeˀ"]
    assert {row["Parameter_ID"] for row in death} == {"m51"}
    assert death[1]["Variant_Of_Key"] == death[0]["Entry_Key"]

    peahen = first["bahl1962korwa:C:c1.p117.r15.i1586"]
    assert peahen["Final_Form"] == "maraːˀ"
    assert peahen["Final_Gloss"] == "peahen."
    assert peahen["Parameter_ID"] == "m81"
    assert peahen["Rau_Gloss"] == "peacock"

    with source.LEGACY_FORMS.open(encoding="utf-8", newline="") as handle:
        legacy = [row for row in csv.reader(handle) if len(row) >= 8 and row[7].startswith("BAHL[")]
    assert len(legacy) == 10
    assert {row[1] for row in legacy} == source.LEGACY_ONLY_PARAMETERS
    assert not ({row[1] for row in legacy} & set(source.PROTO_LINKS.values()))


def test_notes_comparisons_uncertainty_and_locator_policy():
    first = first_audit_rows()
    assert Counter(row["Note_Class"] for row in first.values()) == {
        "none": 1742, "comment": 21, "comparative": 19, "other": 10,
    }
    comparison = first["bahl1962korwa:C:c1.p49.r7.i713"]
    assert comparison["Raw_Note"] == "!H./bhii/.  #07690."
    assert comparison["Final_Etymology"] == "H./bhii/.  #07690."
    assert not comparison["Final_Notes"]
    assert comparison["Tags"].split() == ["part", "emph"]

    query = first["bahl1962korwa:C:c1.p41.r13.i598"]
    assert query["Final_Notes"] == "??.  6460."
    assert query["Tags"] == "uncertain"
    assert not query["Final_Etymology"]

    zero_padded = first["bahl1962korwa:C:c1.p014.r32.i1414"]
    assert zero_padded["Page"] == "014"
    assert zero_padded["Citation"] == "BAHL[p. 14, row 32, item 1414]"
    assert zero_padded["Final_Notes"] == "used only by children.  6260."


def test_manifest_profile_sample_and_offline_rebuild():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 1792
    assert manifest["source_variant_rows"] == 1830
    assert manifest["installed_rows"] == 1830
    assert manifest["proto_munda_linked_source_records"] == 57
    assert manifest["proto_munda_linked_installed_rows"] == 58
    assert manifest["legacy_bahl_rows_replaced"] == 57
    assert set(manifest["legacy_bahl_rows_retained"]) == source.LEGACY_ONLY_PARAMETERS
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
        rows = [
            row for row in all_rows
            if row.get("Entry_Key", "").startswith(source.SOURCE_KEY + ":")
        ]
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
    assert len(rows) == 1830
    assert {row["Language_ID"] for row in rows} == {"kw"}
    assert linked == 58
    assert all(row["Source"].startswith("BAHL[") for row in rows)


SOURCE_KEY = source.SOURCE_KEY
