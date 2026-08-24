import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/buddruss_grangali_1979.py"
SPEC = importlib.util.spec_from_file_location("buddruss_grangali_1979", SCRIPT)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_grangali_ningalami_and_shumashti_are_independent_languages():
    languages = {row["ID"]: row for row in rows(ROOT / "cldf/languages.csv")}
    assert {key: languages[key]["Name"] for key in ("Gng", "Ning", "Shum")} == {
        "Gng": "Grangali",
        "Ning": "Ningalami",
        "Shum": "Shumashti",
    }
    assert languages["Gng"]["Glottocode"] == languages["Ning"]["Glottocode"] == "gran1245"
    dialects = rows(ROOT / "cldf/dialects.csv")
    assert not any(row["Language_ID"] in {"Gng", "Ning"} for row in dialects)


def test_checked_in_transcription_covers_the_whole_atlas_questionnaire():
    records = source.records()
    coverage = set().union(*(source.atlas_numbers(row["atlas"]) for row in records))
    assert coverage == set(range(1, 168))
    assert len(records) == 173
    assert Counter(bool(row["form"]) for row in records) == {True: 170, False: 3}
    assert {row["atlas"] for row in records if not row["form"]} == {"47", "110", "166"}
    comparisons = source.comparison_records()
    assert len(comparisons) == 150
    assert Counter(row["lect"] for row in comparisons) == {
        "Ningalami": 59,
        "Shumashti": 91,
    }


def test_audit_is_complete_and_every_install_has_a_stable_key():
    audit = rows(ROOT / "data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv")
    assert len(audit) == 323
    assert Counter(row["Final_Status"] for row in audit) == {
        "installed_form": 170,
        "installed_comparison": 150,
        "excluded_unattested": 3,
    }
    keys = [row["Emitted_Key"] for row in audit if row["Emitted_Key"]]
    assert len(keys) == len(set(keys)) == 320
    assert all(row["Review"] == "full manual census against the 300 dpi render" for row in audit)
    assert {row["Collation_Date"] for row in audit} == {source.COLLATION_DATE}
    assert all(row["Material_Error"] == "no" for row in audit)


def test_manifest_records_rights_ocr_and_conservative_cdial_policy():
    manifest = json.loads(
        (ROOT / "data/other/forms/raw_data/20260819-buddruss-grangali-manifest.json").read_text()
    )
    assert manifest["pdf_sha256"] == source.PDF_SHA256
    assert manifest["pdf_pages"] == 23
    assert manifest["pdf_redistributed"] is False
    assert manifest["extraction"]["atlas_number_coverage"] == [1, 167]
    assert manifest["outputs"]["form_count"] == 320
    assert manifest["extraction"]["manual_census_count"] == 323
    assert manifest["extraction"]["comparison_lect_counts"] == {
        "Ningalami": 59,
        "Shumashti": 91,
    }
    assert manifest["extraction"]["forms_corrected_after_census"] == 104
    assert manifest["extraction"]["transcription_uncertainties_remaining"] == 0
    assert "only direct" in manifest["scope"]["cdial_policy"]


def test_source_rows_preserve_original_notation_and_only_direct_links():
    with (ROOT / "data/other/forms/20260819-buddruss-grangali.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        forms = list(csv.reader(handle))
    assert len(forms) == 320
    by_key = {row[10]: row for row in forms}
    assert {row[0] for row in forms} == {"Gng", "Ning", "Shum"}
    assert by_key["buddruss-grangali1979:item-1"][0] == "Gng"
    assert by_key["buddruss-grangali1979:item-2:ningalami"][0] == "Ning"
    assert by_key["buddruss-grangali1979:item-2:shumashti"][0] == "Shum"
    assert by_key["buddruss-grangali1979:item-98-99"][2] == "ãc̣"
    assert by_key["buddruss-grangali1979:item-67"][2] == "goā́t"
    assert by_key["buddruss-grangali1979:item-93"][2] == "ǐm"
    assert by_key["buddruss-grangali1979:item-102"][2] == "naṅacə́"
    assert by_key["buddruss-grangali1979:item-131"][1] == "12578"
    assert by_key["buddruss-grangali1979:item-86"][1] == "4251"
    assert {row[1] for row in forms if row[1]} == {"4251", "12578"}
    assert by_key["buddruss-grangali1979:item-23:form-2"][11] == \
        "buddruss-grangali1979:item-23"


def test_compiled_rows_exist_when_cldf_has_been_built():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists():
        return
    compiled = rows(forms_path)
    source_rows = [row for row in compiled if source.SOURCE_ID in row["Source"]]
    # A stale generated CLDF tree may exist before this source's first full build.
    if not source_rows:
        return
    if forms_path.stat().st_mtime < source.FORM_OUTPUT.stat().st_mtime:
        return
    assert len(source_rows) == 320
    assert "�" not in "".join("|".join(row.values()) for row in source_rows)
    identities = rows(ROOT / "data/form-identities.csv")
    eye_id = next(
        row["Form_ID"] for row in identities
        if row["Source_Key"] == f"{source.SOURCE_ID}:item-98-99"
    )
    eye = next(row for row in source_rows if row["ID"] == eye_id)
    assert (eye["Form"], eye["Original"]) == ("ãʦ̣", "ãc̣")
