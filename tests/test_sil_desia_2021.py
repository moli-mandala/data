from __future__ import annotations

import csv
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_desia_2021"
IMPORTER = PACKAGE / "import_desia.py"
MANUAL = PACKAGE / "manual_review.tsv"
PAGES = PACKAGE / "page_review.tsv"
UNRESOLVED = PACKAGE / "unresolved_readings.tsv"
DISCREPANCIES = PACKAGE / "metadata_discrepancies.tsv"
GLYPH_CORRECTIONS = PACKAGE / "glyph_order_corrections.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-desia.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-desia-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-desia-manifest.json"
PROFILE = ROOT / "conversion/sil-desia.txt"


def dict_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def form_rows() -> list[list[str]]:
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_importer_is_reproducible_and_scaffold_is_not_installation_input():
    result = subprocess.run(
        ["python3", str(IMPORTER)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "reviewed_lines=4696 forms=4655 audit_rows=4693"
    source = IMPORTER.read_text(encoding="utf-8")
    assert 'MANUAL = HERE / "manual_review.tsv"' in source
    assert "text_layer_scaffold.txt" not in source
    assert "pdfplumber" not in source


def test_manual_review_and_page_completion_are_explicit():
    rows = dict_rows(MANUAL, "\t")
    pages = dict_rows(PAGES, "\t")
    assert len(rows) == 4696
    assert Counter(row["Manual_Form"] == "no entry" for row in rows) == Counter({False: 4658, True: 38})
    assert {row["Review_Status"] for row in rows} == {"complete"}
    assert {row["Confidence"] for row in rows} == {"high"}
    assert {row["Review_Method"] for row in rows} == {
        "manual visual comparison against rendered source image; embedded text used only as scaffold"
    }
    assert len(pages) == 48
    assert {int(row["PDF_Page"]) for row in pages} == set(range(80, 128))
    assert {row["Review_Status"] for row in pages} == {"complete"}
    assert sum(int(row["Response_Lines"]) for row in pages) == 4696


def test_audit_accounts_for_every_target_cell_form_and_blank():
    rows = dict_rows(AUDIT)
    assert len(rows) == 4693
    assert len({(row["Item"], row["Site"]) for row in rows}) == 210 * 19
    assert Counter(row["Status"] for row in rows) == Counter(installed=4655, missing=38)
    blanks = [row for row in rows if row["Status"] == "missing"]
    assert {(row["Item"], row["Gloss"]) for row in blanks} == {("23", "urine"), ("24", "feces")}
    assert all(row["Review_Status"] == "confirmed-blank" and not row["Manual_Form"] for row in blanks)
    assert all("manual visual comparison" in row["Review_Method"] for row in rows)


def test_installed_rows_preserve_ipa_variants_groups_and_dialect_topology():
    rows = form_rows()
    assert len(rows) == 4655
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"AdivasiOriya"}
    assert len({row[14] for row in rows}) == 19
    assert all(row[2] == row[5] and unicodedata.normalize("NFC", row[2]) == row[2] for row in rows)
    assert all(row[1] == row[4] == row[8] == row[9] == "" for row in rows)
    assert all("non-etymological" in row[6] for row in rows)
    by_key = {row[10]: row for row in rows}
    assert by_key["sildesia2021:i001:potenda:f2"][2] == "ɡaɡɔɖɨ muɳɖ"
    assert "[blank]" in by_key["sildesia2021:i109:ghumar:f1"][6]
    assert by_key["sildesia2021:i113:souraguda:f1"][2] == "ɔɳɖra munus"
    assert "group(s) 1,2" in by_key["sildesia2021:i113:souraguda:f1"][6]
    assert by_key["sildesia2021:i138:konda-maliguda:f1"][2] == "budʒa"
    assert by_key["sildesia2021:i138:patta-maliguda:f1"][2] == "budʒa"


def test_exactly_one_blank_similarity_group_and_three_merged_duplicates():
    manual = dict_rows(MANUAL, "\t")
    blank_groups = [row for row in manual if not row["Similarity_Group"] and row["Manual_Form"] != "no entry"]
    assert [(row["PDF_Page"], row["Item"], row["Site"], row["Manual_Form"]) for row in blank_groups] == [
        ("103", "109", "Ghumar", "apa")
    ]
    audit = dict_rows(AUDIT)
    merged = [row for row in audit if row["Similarity_Groups"] == "1;2"]
    assert {(row["Item"], row["Site"], row["Manual_Form"]) for row in merged} == {
        ("113", "Souraguda", "ɔɳɖra munus"),
        ("138", "Konda Maliguda", "budʒa"),
        ("138", "Patta Maliguda", "budʒa"),
    }


def test_no_unresolved_readings_and_metadata_discrepancies_are_explicit():
    assert dict_rows(UNRESOLVED, "\t") == []
    discrepancies = dict_rows(DISCREPANCIES, "\t")
    assert len(discrepancies) == 7
    assert {row["Topic"] for row in discrepancies} >= {"Dom list", "Dhulia list", "Bonda village"}
    assert all(row["Editorial_Action"] for row in discrepancies)


def test_every_text_layer_combining_mark_misattachment_is_audited():
    rows = dict_rows(GLYPH_CORRECTIONS, "\t")
    assert len(rows) == 542
    assert all(row["Scaffold_Form"] != row["Manual_Form"] for row in rows)
    examples = {(row["Item"], row["Site"]): (row["Scaffold_Form"], row["Manual_Form"]) for row in rows}
    assert examples[("203", "Aunli")] == ("tu̪ i", "t̪ui")
    assert examples[("5", "Gumalput")] == ("ɐk̃ ɪ", "ɐ̃kɪ")
    installed = {row[10]: row for row in form_rows()}
    assert installed["sildesia2021:i203:aunli:f1"][2] == "t̪ui"


def test_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("tʃeɳɖi", column="IPA", segment_separator="", separator="") == "ceṇḍi"
    assert tokenizer("d̪iɔ", column="IPA", segment_separator="", separator="") == "dio"
    assert tokenizer("a:ki", column="IPA", segment_separator="", separator="") == "aːki"
    assert tokenizer("ãːki", column="IPA", segment_separator="", separator="") == "ā̃ki"
    for row in form_rows():
        converted = tokenizer(row[2], column="IPA", segment_separator="", separator="")
        assert "�" not in converted


def test_manifest_pins_source_and_review_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == "04de0004c1375955c1adbeb8941b187aa4fc88f484ee00e9bc69655813e6690b"
    assert manifest["pdf_pages"] == 158 and manifest["pdf_bytes"] == 3737879
    assert manifest["conceptual_cells"] == 3990
    assert manifest["manually_reviewed_response_lines"] == 4696
    assert manifest["manually_reviewed_attested_response_lines"] == 4658
    assert manifest["manually_reviewed_blank_cells"] == 38
    assert manifest["installed_forms"] == 4655
    assert manifest["unresolved_readings"] == 0
    assert manifest["text_layer_glyph_order_corrections"] == 542
    assert manifest["ocr_heavy_addendum"].startswith("not applicable")
