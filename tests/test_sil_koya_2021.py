"""Focused checks for the manually transcribed SIL Koya survey wordlists."""

import csv
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_koya_2021"
INSTALLED = ROOT / "data/other/forms/20260828-sil-koya.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-koya-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-koya-manifest.json"
PROFILE = ROOT / "conversion/sil-koya.txt"
SOURCE_KEY = "devagnanavaram-et-al2021koya"
TARGET_SITES = {"JAG", "CHI", "POD", "UTN", "BHG", "BHM", "MAL"}

sys.path.insert(0, str(PACKAGE))
import import_koya  # noqa: E402

ROWS = import_koya.ROWS


def installed_rows():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audit_rows():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_manual_ledger_has_exact_source_topology():
    assert len(ROWS) == 95
    assert sum(len(row["Forms"]) for row in ROWS) == 1840
    assert all(row["Review"] == "manual-source-image" for row in ROWS)
    assert {(row["PDF_Page"], row["Site"]) for row in ROWS} == set(
        import_koya.expected_page_item_pairs()
    )
    assert all(
        len(row["Forms"]) == (10 if row["First_Item"] == 201 else 20)
        for row in ROWS
    )


def test_installed_counts_shape_and_stable_keys():
    rows = installed_rows()
    assert len(rows) == 1438
    assert {len(row) for row in rows} == {15}
    assert all(row[0] == "Gondi" and row[1] == "" for row in rows)
    assert all(row[2] and row[2] == row[5] for row in rows)
    assert all(row[8] == row[9] == "" for row in rows)
    assert all(row[10].startswith("silkoya1985:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(unicodedata.normalize("NFC", row[2]) == row[2] for row in rows)


def test_importer_rebuild_is_byte_equivalent_at_the_row_level():
    rebuilt, rebuilt_audit, _ = import_koya.build()
    assert rebuilt == installed_rows()
    assert [
        {key: str(value) for key, value in row.items()} for row in rebuilt_audit
    ] == audit_rows()


def test_every_conceptual_slot_is_audited_and_manually_reviewed():
    rows = audit_rows()
    assert len(rows) == 1890
    assert Counter(row["Site_Code"] for row in rows) == {
        code: 210 for code in import_koya.SITES
    }
    assert Counter(row["Status"] for row in rows) == {
        "installed": 1401,
        "missing": 69,
        "excluded": 420,
    }
    assert all(row["Manual_Review"] == "manual-source-image" for row in rows)
    assert all(row["OCR_Evidence"].startswith("tesseract_raw.txt#pdf") for row in rows)
    assert all(row["Reason"] == "excluded comparison control" for row in rows if row["Status"] == "excluded")


def test_blanks_omissions_controls_and_uncertainty_are_explicit():
    rows = audit_rows()
    controls = [row for row in rows if row["Site_Code"] in {"TEL", "ORI"}]
    assert len(controls) == 420 and not any(row["Entry_Keys"] for row in controls)
    mal_gap = [row for row in rows if row["Site_Code"] == "MAL" and 61 <= int(row["Item"]) <= 80]
    assert len(mal_gap) == 20
    assert all("explicit editor note" in row["Reason"] for row in mal_gap)
    east_pronouns = [
        row for row in rows
        if row["Site_Code"] in {"JAG", "CHI", "POD"} and int(row["Item"]) >= 201
    ]
    assert len(east_pronouns) == 30
    assert all("list ends at item 200" in row["Reason"] for row in east_pronouns)
    clipped = next(row for row in rows if row["Site_Code"] == "MAL" and row["Item"] == "200")
    assert clipped["Status"] == "missing"
    assert clipped["Uncertainty"] == "item 200 absent/clipped between source pages"
    ambiguous = next(row for row in rows if row["Site_Code"] == "MAL" and row["Item"] == "31")
    assert "illegible medial" in ambiguous["Uncertainty"]


def test_no_ocr_only_or_control_record_is_installed():
    audit = audit_rows()
    keys = {key for row in audit if row["Status"] == "installed" for key in row["Entry_Keys"].split("|")}
    installed = installed_rows()
    assert keys == {row[10] for row in installed}
    assert all("manual source-image transcription" in row[6] for row in installed)
    assert not any("OCR" in row[6] for row in installed)
    assert not any(row[0] in {"Telugu", "Oriya"} for row in installed)


def test_slash_variants_split_but_parenthetical_material_is_preserved():
    rows = installed_rows()
    by_key = {row[10]: row for row in rows}
    assert by_key["silkoya1985:jag:i037:v1"][2] == "daɾam"
    assert by_key["silkoya1985:jag:i037:v2"][2] == "nol"
    assert by_key["silkoya1985:utn:i079:v1"][2] == "(pul)ɡɦobi"
    assert not any("/" in row[2] for row in rows)


def test_seven_distinct_dialect_tags_and_representative_forms():
    rows = installed_rows()
    tags = {row[14].split()[0] for row in rows}
    assert len(tags) == 7
    assert any("bhamani-gondi" in tag for tag in tags)
    assert any("bhamani-madia" in tag for tag in tags)
    by_key = {row[10]: row for row in rows}
    assert by_key["silkoya1985:jag:i001:v1"][2] == "oleu"
    assert by_key["silkoya1985:chi:i021:v1"][2] == "gundikaia"
    assert by_key["silkoya1985:utn:i001:v1"][2] == "mɛːnd̪ol"
    assert by_key["silkoya1985:mal:i021:v1"][2] == "dʒiːva"


def test_profile_covers_all_installed_forms():
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed_rows():
        assert "�" not in tokenizer(row[2], column="IPA")


def test_shared_profile_routing_and_metadata_registration():
    sys.path.insert(0, str(ROOT))
    from make_cldf import parse_file

    errors = io.StringIO()
    parsed, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 1438, "for_conversion": 1438}
    assert errors.getvalue() == ""
    assert len(parsed) == 1438
    assert all(row.ipa == row.old_form and "�" not in row.form for row in parsed)

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith("sil-koya-1985-")
        ]
    assert len(dialects) == 7
    assert {row["Language_ID"] for row in dialects} == {"Gondi"}
    assert {row["Source_Language_ID"] for row in dialects} == TARGET_SITES
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all(row["Quality"] == "C" for row in dialects)
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    assert "No installed form originates from OCR" in bib.split(
        f"@techreport{{{SOURCE_KEY},", 1
    )[1].split("\n}", 1)[0]


def test_manifest_pins_source_and_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == "a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d"
    assert manifest["source_pdf_pages"] == 124
    assert manifest["counts"]["conceptual_list_slots"] == 1890
    assert manifest["counts"]["source_image_cells_manually_reviewed"] == 1840
    assert manifest["counts"]["omitted_slots_accounted_for"] == 50
    assert manifest["policy"]["ocr"].startswith("comparison scaffold only")
