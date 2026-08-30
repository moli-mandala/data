from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_noira_2015"
INSTALLED = ROOT / "data/other/forms/20260829-sil-noira.csv"
PROFILE = ROOT / "conversion/sil-noira.txt"
STAGED_SHA256 = "c82983a319d6d6fbf5c07063f0655ae3e4e8e3890d625e1bfc2a38f95c811746"
PROFILE_SHA256 = "3932523f127f4a13a94915dbd88bc21d2cac5867138bec7f2ce03a061e7f0de5"


def dict_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def test_noira_installed_target_rows_are_exact_and_immutable() -> None:
    assert INSTALLED.read_bytes() == (PACKAGE / "staged_forms.csv").read_bytes()
    assert hashlib.sha256(INSTALLED.read_bytes()).hexdigest() == STAGED_SHA256
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    assert len(forms) == 2714
    assert Counter(map(len, forms)) == Counter({15: 2714})
    assert len({row[10] for row in forms}) == 2714
    assert Counter(row[0] for row in forms) == Counter(
        Noiri=1248, DungraBhili=524, ko=476, Ni=234, Goj=232
    )
    assert Counter(row[10].split(":")[-2] for row in forms) == Counter(
        DBM=273, NPN=269, NCH=258, NGO=256, DBA=251, TKO=247,
        KNA=239, NJA=234, GTA=232, NTE=229, KTA=226,
    )
    assert {row[7].split("[", 1)[0] for row in forms} == {"varghesekumar2015noira"}
    assert all(row[10].startswith("noira2015:p") for row in forms)
    assert not any(f"list {site}]" in row[7] for row in forms for site in ["NAS", "BMU", "NTO", "GUJ", "MAR", "HIN"])


def test_noira_all_exclusions_remain_audit_only() -> None:
    audit = dict_rows(PACKAGE / "exhaustive_audit.tsv", "\t")
    assert len(audit) == 3570
    assert Counter(row["Scope"] for row in audit) == Counter(
        new_target=2310, republished_dhule=630, comparison_control=630
    )
    assert Counter(row["Review_Status"] for row in audit) == Counter(
        attested=3526, source_blank=44
    )
    assert sum(int(row["Installed_Count"]) for row in audit) == 2714
    assert all(
        row["Installed_Count"] == "0" and not row["Entry_Keys"]
        for row in audit if row["Scope"] != "new_target"
    )
    reconciliation = dict_rows(PACKAGE / "dhule_republication_reconciliation.tsv", "\t")
    assert len(reconciliation) == 630
    assert Counter(row["Noira_Site"] for row in reconciliation) == Counter(
        NAS=210, BMU=210, NTO=210
    )
    assert Counter(row["Comparison"] for row in reconciliation) == Counter({
        "literal-ledger-exact": 3,
        "same-source-representation-differs": 627,
    })
    assert not dict_rows(PACKAGE / "unresolved_readings.tsv", "\t")


def test_noira_profile_is_exact_and_routed_by_source_key() -> None:
    assert PROFILE.read_bytes() == (PACKAGE / "conversion_profile.tsv").read_bytes()
    assert hashlib.sha256(PROFILE.read_bytes()).hexdigest() == PROFILE_SHA256
    inventory = dict_rows(PACKAGE / "profile_inventory.tsv", "\t")
    assert len(inventory) == 54
    assert all(row["Present_In_Staged_Targets"] == "yes" for row in inventory)
    build = (ROOT / "make_cldf.py").read_text(encoding="utf-8")
    assert '"sil-noira",' in build
    assert 'if source_key == "varghesekumar2015noira":' in build
    assert 'row_ipa = "sil-noira"' in build
    assert "row_convert = True" in build


def test_noira_language_and_dialect_metadata_match_installed_tags() -> None:
    languages = {row["ID"]: row for row in dict_rows(ROOT / "cldf/languages.csv")}
    assert languages["DungraBhili"]["Glottocode"] == "dung1251"
    assert languages["DungraBhili"]["Latitude"] == languages["DungraBhili"]["Longitude"] == ""
    assert languages["Noiri"]["Glottocode"] == "noir1238"
    assert "Kotli routing is provisional" in languages["Noiri"]["Location"]

    dialects = {row["ID"]: row for row in dict_rows(ROOT / "cldf/dialects.csv")}
    registry = {row["Site_Code"]: row for row in dict_rows(PACKAGE / "list_registry.tsv", "\t")}
    expected = {row["Dialect_ID"]: row["Language_ID"] for row in registry.values() if row["Scope"] == "new_target"}
    assert len(expected) == 11
    for dialect_id, language_id in expected.items():
        assert dialects[dialect_id]["Language_ID"] == language_id
        assert dialects[dialect_id]["Latitude"] == dialects[dialect_id]["Longitude"] == ""
    for dialect_id in [
        "sil-noira-2015-kotli-narayanpur",
        "sil-noira-2015-kotli-taradi",
    ]:
        assert dialects[dialect_id]["Language_ID"] == "Noiri"
        assert dialects[dialect_id]["Glottocode"] == ""
        assert "provisional source-supported Noiri routing" in dialects[dialect_id]["Location"]
        assert "historical Kotali/Khandesi" in dialects[dialect_id]["Location"]

    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    installed_tags = {row[14] for row in forms}
    assert installed_tags == {dialects[dialect_id]["Tag"] for dialect_id in expected}


def test_noira_reference_metadata_and_manual_only_policy_are_exact() -> None:
    references = {row["ID"]: row for row in dict_rows(ROOT / "cldf/references.csv")}
    reference = references["varghesekumar2015noira"]
    assert reference["Short"] == "V2015b"
    assert reference["Progress"].startswith(
        "Appendix A3, printed pages 27--72: 2,714 manually verified lexical attestations"
    )
    assert "630 conceptual cells / 834 responses" in reference["Progress"]
    assert "630 cells / 837 responses" in reference["Progress"]
    assert reference["OCR"] == "No"
    assert reference["Etymology_Provenance"] == "none"
    assert "Canonical 96-page PDF SHA-256" in reference["Provenance"]
    bibliography = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert "@techreport{varghesekumar2015noira," in bibliography
    assert "Every retained IPA form was transcribed and visually verified by hand" in bibliography
    assert "supplied or verified no retained reading" in bibliography


def test_noira_shared_integration_manifest_records_only_deferred_global_gates() -> None:
    manifest = json.loads((PACKAGE / "shared_integration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "shared-source-specific-integration-complete"
    assert manifest["installed_target_forms"]["rows"] == 2714
    assert manifest["installed_target_forms"]["unique_entry_keys"] == 2714
    assert manifest["installed_target_forms"]["sha256"] == STAGED_SHA256
    assert manifest["audit_only"]["republished_dhule"]["conceptual_cells"] == 630
    assert manifest["audit_only"]["comparison_controls"]["conceptual_cells"] == 630
    assert manifest["source_local_contract"]["source_blank_cells"] == 44
    assert manifest["source_local_contract"]["unresolved_coordinates"] == []
    assert manifest["metadata"]["new_parent_languages"] == ["DungraBhili"]
    assert manifest["metadata"]["dialects_with_blank_coordinates"] == 11
    assert manifest["deferred"] == [
        "consolidated CLDF/full build and compiled-row survival check",
        "global source-audit/checklist regeneration",
        "full pytest and graph validation",
        "browser database refresh and representative-entry QA",
        "commit and shipping",
    ]
