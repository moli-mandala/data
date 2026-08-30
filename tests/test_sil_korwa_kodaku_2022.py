from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_korwa_kodaku_2022"
IMPORTER = PACKAGE / "import_korwa_kodaku.py"
MANUAL = PACKAGE / "manual_review.tsv"
PAGES = PACKAGE / "page_review.tsv"
UNRESOLVED = PACKAGE / "unresolved_source_codes.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-korwa-kodaku.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-korwa-kodaku-manifest.json"
PROFILE = ROOT / "conversion/sil-korwa-kodaku.txt"
SOURCE_KEY = "behera2022korwakodaku"
TARGET_CODES = set("CDGHKLMRZSbcdjmptw")


def dict_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def form_rows() -> list[list[str]]:
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_importer_is_reproducible_and_does_not_parse_scaffolds():
    result = subprocess.run(
        ["python3", str(IMPORTER)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == (
        "response_lines=2900 cells=5250 installed=4458 audit=6135 unresolved_codes=2"
    )
    source = IMPORTER.read_text(encoding="utf-8")
    assert "manual_review.tsv" in source
    assert "TEXT_SCAFFOLD.read_text" not in source
    assert "OCR.read_text" not in source


def test_manual_review_and_page_completion_are_explicit():
    rows = dict_rows(MANUAL, "\t")
    pages = dict_rows(PAGES, "\t")
    assert len(rows) == 2900
    assert {row["Review_Status"] for row in rows} == {"complete"}
    assert {row["Confidence"] for row in rows} == {"high"}
    assert len(pages) == 25
    assert {int(row["PDF_Page"]) for row in pages} == set(range(66, 91))
    assert {row["Review_Status"] for row in pages} == {"complete"}
    assert sum(int(row["Response_Lines"]) for row in pages) == 2900


def test_audit_accounts_for_every_target_control_and_blank_cell():
    rows = dict_rows(AUDIT)
    assert len(rows) == 6135
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 210 * 25
    assert Counter(row["Status"] for row in rows) == Counter(
        installed=4457, excluded=1628, missing=50
    )
    blanks = [row for row in rows if not row["Manual_Form"]]
    assert len(blanks) == 67
    assert Counter(row["Reason"] for row in blanks) == Counter({
        "source prints NO ENTRY for the entire item": 50,
        "source explicitly assigns NO ENTRY to this site": 4,
        "no response printed for this site/item in the compressed table": 13,
    })
    assert all("manual visual review" in row["Review_Method"] for row in rows)


def test_unidentified_codes_are_transcribed_but_never_guessed_or_installed():
    rows = dict_rows(UNRESOLVED, "\t")
    assert [(row["PDF_Page"], row["Item"], row["Manual_Form"], row["Unknown_Site_Code"]) for row in rows] == [
        ("73", "83", "buluŋg", "u"),
        ("84", "173", "nɐʔa", "n"),
    ]
    assert all("not reassigned" in row["Resolution"] for row in rows)
    assert not any(row[10].endswith(":u:f1") or row[10].endswith(":n:f1") for row in form_rows())


def test_installed_rows_preserve_diacritics_variants_and_non_etymological_groups():
    rows = form_rows()
    assert len(rows) == 4458
    assert {len(row) for row in rows} == {15}
    assert Counter(row[0] for row in rows) == Counter(kw=2160, Kodaku=2298)
    assert len({row[14] for row in rows}) == 18
    assert all(row[2] == row[5] and unicodedata.normalize("NFC", row[2]) == row[2] for row in rows)
    assert all(row[1] == row[4] == row[8] == row[9] == "" for row in rows)
    assert all("non-etymological" in row[6] for row in rows)
    by_key = {row[10]: row for row in rows}
    assert by_key["silkorwakodaku2022:i001:C:f1"][2] == "d̪ẽh"
    assert by_key["silkorwakodaku2022:i001:S:f1"][2] == "d̪ejaɲ"
    assert by_key["silkorwakodaku2022:i104:c:v1"][2] == "koda hɔpoɲ"
    assert by_key["silkorwakodaku2022:i104:c:v2"][2] == "koɖi hɔpoɲ"
    assert not any("/" in row[2] for row in rows)


def test_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in form_rows():
        converted = tokenizer(row[2], column="IPA", segment_separator="", separator="")
        assert "�" not in converted


def test_shared_profile_language_dialect_and_bibliographic_registration():
    sys.path.insert(0, str(ROOT))
    from make_cldf import parse_file

    errors = io.StringIO()
    parsed, stats = parse_file(str(INSTALLED), errors)
    assert stats == {"converted": 4458, "for_conversion": 4458}
    assert errors.getvalue() == ""
    assert len(parsed) == 4458
    assert all(row.ipa == row.old_form and "�" not in row.form for row in parsed)

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["kw"]["Glottocode"] == "korw1242"
    assert languages["Kodaku"]["Glottocode"] == "koda1256"
    assert "ISO 639-3 ksz" in languages["Kodaku"]["Location"]

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = [
            row for row in csv.DictReader(stream)
            if row["ID"].startswith(("sil-korwa-200", "sil-kodaku-200"))
        ]
    assert len(dialects) == 18
    assert Counter(row["Language_ID"] for row in dialects) == Counter(kw=9, Kodaku=9)
    assert {row["Source_Language_ID"] for row in dialects} == TARGET_CODES
    assert all(not row["Latitude"] and not row["Longitude"] for row in dialects)
    assert all(row["Quality"] == "C" for row in dialects)

    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    record = bib.split(f"@techreport{{{SOURCE_KEY},", 1)[1].split("\n}", 1)[0]
    assert "every one of the 2,900 printed response rows was visually checked" in record
    assert "No ambiguous, clipped, illegible, or OCR-derived form is installed" in record


def test_manifest_pins_source_and_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == "a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c"
    assert manifest["pdf_pages"] == 115 and manifest["pdf_bytes"] == 2198621
    counts = manifest["counts"]
    assert counts["conceptual_cells_manually_audited"] == 5250
    assert counts["target_cells_manually_audited"] == 3780
    assert counts["target_attested_cells"] == 3730
    assert counts["target_blank_or_unlisted_cells"] == 50
    assert counts["installed_target_rows_after_slash_expansion"] == 4458
    assert counts["ambiguous_or_illegible_installed_forms"] == 0
    assert manifest["policy"]["ocr"].startswith("comparison scaffold only")
