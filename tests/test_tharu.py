import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
BRACKETED_NUMBER = re.compile(r"^\(\d+\)$")
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_western_tharu_2017"
INSTALLED = ROOT / "data/other/forms/20230530-tharu2.csv"
IMPORTER = SOURCE_DIR / "import_western_tharu_2017.py"
AUDIT = SOURCE_DIR / "staged_audit.tsv"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SITE_IDS = {
    "Tharu-BNM", "Tharu-BNT", "Tharu-RNK", "Tharu-RNS-Sisaikhara",
    "Tharu-RNS-Sisana", "Tharu-RkM", "Tharu-RKB", "Tharu-TkN",
    "Tharu-KkP", "Tharu-SkP", "Tharu-DKS", "Tharu-DDK", "Tharu-DGC",
    "Tharu-DkR", "Tharu-CCC",
}
THARU_LANGUAGE_IDS = {
    "Rana",
    "Dang",
    "Chitwan",
    "Morang",
    "Buksa",
    "Tharu-BNT",
    "Tharu-RNK",
    "Tharu-RNS",
    "Tharu-RkM",
    "Tharu-RKB",
    "Kathoriya",
    "Sunha",
    "Tharu-DKS",
    "Tharu-DGC",
    "Tharu-DkR",
}


def test_webster_tharu_excludes_bracketed_reference_numbers():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows
    assert not any(BRACKETED_NUMBER.fullmatch(row[2]) for row in rows)


def installed_forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def test_western_tharu_importer_rebuilds_frozen_integration_deterministically():
    result = subprocess.run(
        [sys.executable, str(IMPORTER), "--stage", "--install"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["integration"] == {
        "hindi_control_cells_excluded": 210,
        "installed_rows": 3560,
        "installed_sha256": "a8c2012744f6135a93a4f6c3136fd01db2b6795290b8b55a4429154e877fc8c4",
        "rns_uncertain_conceptual_cells": 420,
        "rns_uncertain_installed_forms": 486,
        "staged_audit_rows": 3360,
        "staged_audit_sha256": "70f72321b50908a94305f5cf584d27a5aabb6e324eeadede1dd7ea0262eefd03",
        "target_blank_cells_excluded": 98,
        "unique_entry_keys": 3560,
    }
    assert payload["legacy_reconciliation"] == {
        "manual_target_occurrences": 3560,
        "legacy_target_occurrences": 3548,
        "exact_occurrences": 2794,
        "manual_only_occurrences": 766,
        "legacy_only_occurrences": 754,
    }


def test_western_tharu_rich_forms_scope_keys_and_conservative_graph_contract():
    rows = installed_forms()
    assert len(rows) == 3560
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == 3560
    assert Counter(row["Language_ID"] for row in rows) == Counter({
        "Rana": 1442, "Dang": 984, "Buksa": 490,
        "Kathoriya": 229, "Sunha": 229, "Chitwan": 186,
    })
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(row["Parameter_ID"] == row["Native"] == "" for row in rows)
    assert all(
        row[field] == ""
        for row in rows
        for field in (
            "Cognateset", "Etymology", "Variant_Of_Key", "Borrowed_From_Key",
            "Derivation_Parent_Keys",
        )
    )
    assert all(row["Source"].startswith("webster[Appendix B, printed p. ") for row in rows)
    assert all("not cognacy" in row["Notes"] for row in rows if "group(s)" in row["Notes"])


def test_western_tharu_audit_preserves_all_rns_identity_uncertainties_and_exclusions():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    assert len(audit) == 3360
    assert Counter(row["Disposition"] for row in audit) == Counter(installed=3052, excluded=308)
    assert sum(int(row["Installed_Count"]) for row in audit) == 3560
    rns = [row for row in audit if row["Site_Key"].startswith("RNS_")]
    assert len(rns) == 420
    assert all(row["Site_Assignment_Confidence"] == "medium" for row in rns)
    assert all("duplicate source code RNS" in row["Uncertainty"] for row in rns)
    forms = installed_forms()
    uncertain = [row for row in forms if "uncertain" in row["Tags"].split()]
    assert len(uncertain) == 486
    assert all("RNS occurrence" in row["Source"] for row in uncertain)
    assert all("dialect assignment uncertain" in row["Notes"] for row in uncertain)


def test_western_tharu_sites_reference_profile_and_checklist_are_integrated():
    rows = installed_forms()
    assert {row["Tags"].split(":")[2] for row in rows} == SITE_IDS
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    assert SITE_IDS <= dialects.keys()
    assert dialects["Tharu-RNS-Sisaikhara"]["Language_ID"] == "Rana"
    assert dialects["Tharu-RNS-Sisana"]["Language_ID"] == "Rana"
    assert not dialects["Tharu-RNS-Sisaikhara"]["Latitude"]
    assert dialects["Tharu-CCC"]["Language_ID"] == "Chitwan"
    assert "source label Thakur Tharu" in dialects["Tharu-TkN"]["Location"]
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert "@techreport{webster," in bib
    assert "ocr          = {No}" in bib
    assert "a060adebc3c7508b541522ac19b9b7d068ae9ca59c04f7f8e1078eab09e0486c" in bib
    assert (ROOT / "conversion/sil-western-tharu.txt").exists()
    assert "'tharu2': 'sil-western-tharu'" in (ROOT / "utils.py").read_text(encoding="utf-8")
    make_cldf = (ROOT / "make_cldf.py").read_text(encoding="utf-8")
    assert 'if source_key == "webster":' in make_cldf
    assert 'row_ipa = "sil-western-tharu"' in make_cldf
    audit_registry = (ROOT / "audit_source_ingestions.py").read_text(encoding="utf-8")
    assert '"20230530-tharu2": {' in audit_registry
    assert '"profiles": ["conversion/sil-western-tharu.txt"]' in audit_registry
    checklist = (ROOT / "source_checklists/20230530-tharu2.md").read_text(encoding="utf-8")
    assert "Installed rows: 3560" in checklist
    assert "all 420 conceptual cells" in checklist


def test_compiled_tharu_excludes_bracketed_reference_numbers():
    source = ROOT / "cldf/forms.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = (
            row
            for row in csv.DictReader(stream)
            if row["Language_ID"] in THARU_LANGUAGE_IDS
        )
        assert not any(BRACKETED_NUMBER.fullmatch(row["Form"]) for row in rows)
