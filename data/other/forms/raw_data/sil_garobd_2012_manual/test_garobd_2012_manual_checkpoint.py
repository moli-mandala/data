from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHUNKS = HERE / "manual_chunks"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manual_line_and_cell_ledgers_are_exhaustive_ocr_blind_and_nfc():
    manual = load_module("garobd_manual_001_005", CHUNKS / "hand_keyed_items_001_005.py")
    lines = manual.line_rows()
    cells = manual.cell_rows(lines)
    assert len(lines) == 47
    assert len(cells) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in cells} == {
        (str(item), code) for item in range(1, 6) for code in manual.SITE_CODES
    }
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("52", "45") for row in lines + cells)
    assert len({row["Line_ID"] for row in lines}) == 47
    assert all(len(set(row["Bracket_Codes"])) == len(row["Bracket_Codes"]) for row in lines)
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + cells)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + cells)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + cells for value in row.values())


def test_checkpoint_counts_blanks_scopes_and_variants_are_exact():
    rows = read_tsv(CHUNKS / "items_001_005_cells.tsv")
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 95
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Scope"] == ("control_audit_only" if row["Site_Code"] == "0" else "neutral_unreconciled") for row in rows)
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("1", "a")]["Manual_Transcription"] == "sɨlɡa"
    assert by_cell[("2", "j")]["Manual_Transcription"] == "jao̯ bri"
    assert by_cell[("2", "p")]["Manual_Transcription"] == "sɨŋŋei̯"
    assert by_cell[("3", "g")]["Manual_Transcription"] == "dʒonakʼ | dʒonakʼ"
    assert by_cell[("3", "g")]["Similarity_Groups"] == "3|6"
    assert by_cell[("3", "m")]["Manual_Transcription"] == "tʃaŋ ɨi̯"
    assert by_cell[("4", "i")]["Manual_Transcription"] == "askʰi | askʰi"
    assert by_cell[("4", "i")]["Similarity_Groups"] == "1|3"
    assert by_cell[("4", "j")]["Manual_Transcription"] == "kʰlor"
    blank = by_cell[("5", "p")]
    assert blank["Manual_Transcription"] == ""
    assert blank["Similarity_Groups"] == "0"
    assert blank["Source_Qualification"] == 'printed "no entry"'


def test_second_manual_block_is_exhaustive_and_preserves_repetitions():
    manual = load_module("garobd_manual_006_010", CHUNKS / "hand_keyed_items_006_010.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_006_010_cells.tsv")
    assert generated == rows
    assert len(lines) == 41
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(6, 11) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 76,
        "source_blank": 9,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 79
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("7", "n")]["Manual_Transcription"] == "ramdʰonukʼ | rɔŋdʰɔnu"
    assert by_cell[("7", "n")]["Similarity_Groups"] == "4|4"
    assert by_cell[("8", "j")]["Manual_Transcription"] == "lɨŋ ɪr"
    assert by_cell[("8", "p")]["Manual_Transcription"] == "lɨŋ ir"
    assert by_cell[("9", "a")]["Manual_Transcription"] == "(mɨkʼkʰa) hɛlapuŋa"
    assert by_cell[("10", "e")]["Manual_Transcription"] == "kʰum prɛʔta | kʰum prɛʔta"
    assert by_cell[("10", "e")]["Similarity_Groups"] == "1|2"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {
        ("7", "f"), ("7", "l"), ("7", "o"), ("7", "p"),
        ("9", "f"), ("9", "n"),
        ("10", "b"), ("10", "o"), ("10", "p"),
    }


def test_third_manual_block_preserves_source_conflict_and_small_marks():
    manual = load_module("garobd_manual_011_015", CHUNKS / "hand_keyed_items_011_015.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_011_015_cells.tsv")
    assert generated == rows
    assert len(lines) == 34
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(11, 16) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 79,
        "source_blank": 5,
        "source_conflict": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 80
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    conflict = by_cell[("12", "p")]
    assert conflict["Manual_Transcription"] == "dɔm"
    assert conflict["Similarity_Groups"] == "6"
    assert conflict["Source_Line_IDs"] == "i012-l01|i012-l10"
    assert conflict["Source_Qualification"] == 'also printed "no entry" in group 0'
    assert conflict["Review_Status"] == "source_conflict"
    assert "both group-0 no entry" in conflict["Uncertainty"]
    assert by_cell[("11", "p")]["Manual_Transcription"] == "duriou̯"
    assert by_cell[("13", "l")]["Manual_Transcription"] == "tɨi̯"
    assert by_cell[("15", "p")]["Manual_Transcription"] == "kmia̯n"
    assert by_cell[("15", "0")]["Manual_Transcription"] == "mat̪i"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("11", "a"), ("11", "f"), ("11", "h"), ("11", "n"), ("12", "l")}


def test_fourth_manual_block_preserves_repeats_underties_and_retroflexion():
    manual = load_module("garobd_manual_016_020", CHUNKS / "hand_keyed_items_016_020.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_016_020_cells.tsv")
    assert generated == rows
    assert len(lines) == 49
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(16, 21) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 99
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("16", "i")]["Manual_Transcription"] == "hadɨbɛkʼ | hadɨbɛkʼ"
    assert by_cell[("16", "i")]["Similarity_Groups"] == "2|3"
    assert by_cell[("16", "b")]["Manual_Transcription"] == "haʔdilɛka / kadoŋ | haʔdilɛka / kadoŋ"
    assert by_cell[("17", "f")]["Review_Status"] == "source_blank"
    assert by_cell[("18", "b")]["Manual_Transcription"] == "loŋtʰai̯"
    assert by_cell[("19", "a")]["Manual_Transcription"] == "haŋtʃʰɛŋ | haŋtʃʰɛŋ"
    assert by_cell[("19", "j")]["Manual_Transcription"] == "ɖʒia̯p"
    assert by_cell[("19", "p")]["Manual_Transcription"] == "ɖʒmia̯k"
    assert by_cell[("20", "0")]["Manual_Transcription"] == "ʃona"


def test_fifth_manual_block_preserves_underties_group_a_and_exact_blanks():
    manual = load_module("garobd_manual_021_025", CHUNKS / "hand_keyed_items_021_025.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_021_025_cells.tsv")
    assert generated == rows
    assert len(lines) == 41
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(21, 26) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 81,
        "source_blank": 4,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 81
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("21", "0")]["Manual_Transcription"] == "rupa"
    assert by_cell[("22", "b")]["Manual_Transcription"] == "tai̯ni"
    assert by_cell[("22", "j")]["Manual_Transcription"] == "hɨnta"
    assert by_cell[("23", "i")]["Manual_Transcription"] == "midʒao̯"
    assert by_cell[("23", "b")]["Manual_Transcription"] == "mui̯ja"
    assert by_cell[("23", "0")]["Manual_Transcription"] == "gotokal / kalke"
    assert by_cell[("24", "n")]["Manual_Transcription"] == "ɖʒɛlo"
    assert by_cell[("24", "j")]["Manual_Transcription"] == "ɨrtip"
    assert by_cell[("24", "j")]["Similarity_Groups"] == "A"
    assert by_cell[("25", "e")]["Manual_Transcription"] == "hatʰai̯"
    assert by_cell[("25", "k")]["Manual_Transcription"] == "tiu̯"
    assert by_cell[("25", "0")]["Manual_Transcription"] == "ʃɔpta"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("21", "p"), ("23", "l"), ("24", "l"), ("25", "a")}


def test_sixth_manual_block_preserves_item_28_repetitions_and_small_marks():
    manual = load_module("garobd_manual_026_030", CHUNKS / "hand_keyed_items_026_030.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_026_030_cells.tsv")
    assert generated == rows
    assert len(lines) == 31
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(26, 31) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 92
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("26", "a")]["Manual_Transcription"] == "ɖʒa"
    assert by_cell[("27", "j")]["Manual_Transcription"] == "snɪm"
    assert by_cell[("28", "a")]["Manual_Transcription"] == "ʃal | ʃal"
    assert by_cell[("28", "a")]["Similarity_Groups"] == "1|2"
    assert by_cell[("28", "k")]["Manual_Transcription"] == "sɨŋɨi̯ | sɨŋɨi̯"
    assert by_cell[("28", "k")]["Similarity_Groups"] == "4|6"
    assert by_cell[("28", "p")]["Manual_Transcription"] == "sɨŋ ŋei̯"
    assert by_cell[("28", "0")]["Manual_Transcription"] == "dɪn"
    assert by_cell[("29", "g")]["Manual_Transcription"] == "pʰrɨŋ"
    assert by_cell[("29", "b")]["Manual_Transcription"] == "manatʼ"
    assert by_cell[("30", "i")]["Manual_Transcription"] == "ʃalɖʒatʼtʃi"
    assert by_cell[("30", "a")]["Manual_Transcription"] == "ʃalɖʒatʼtʰi"
    assert by_cell[("30", "j")]["Manual_Transcription"] == "bri pɨndɨŋ"
    blank = by_cell[("30", "p")]
    assert blank["Review_Status"] == "source_blank"
    assert blank["Manual_Transcription"] == ""


def test_seventh_manual_block_preserves_repeated_sites_and_rice_vowels():
    manual = load_module("garobd_manual_031_035", CHUNKS / "hand_keyed_items_031_035.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_031_035_cells.tsv")
    assert generated == rows
    assert len(lines) == 32
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(31, 36) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 87
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("31", "p")]["Manual_Transcription"] == "ɖʒanmot"
    assert by_cell[("32", "e")]["Manual_Transcription"] == "wala | wala"
    assert by_cell[("32", "e")]["Similarity_Groups"] == "1|4"
    assert by_cell[("32", "b")]["Manual_Transcription"] == "pʰarokʼ"
    assert by_cell[("33", "b")]["Manual_Transcription"] == "mai̯"
    assert by_cell[("33", "j")]["Manual_Transcription"] == "ɖʒiba"
    assert by_cell[("34", "b")]["Manual_Transcription"] == "mai̯ruŋ"
    assert by_cell[("34", "m")]["Manual_Transcription"] == "mai̯roŋ"
    assert by_cell[("34", "j")]["Manual_Transcription"] == "kʰao̯"
    assert by_cell[("35", "m")]["Manual_Transcription"] == "mai̯ | mai̯mɨn"
    assert by_cell[("35", "m")]["Similarity_Groups"] == "2|4"
    assert by_cell[("35", "0")]["Manual_Transcription"] == "bʰat"


def test_eighth_manual_block_preserves_corn_potato_variants_and_blanks():
    manual = load_module("garobd_manual_036_040", CHUNKS / "hand_keyed_items_036_040.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_036_040_cells.tsv")
    assert generated == rows
    assert len(lines) == 23
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(36, 41) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 82,
        "source_blank": 3,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 85
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("36", "0")]["Manual_Transcription"] == "gɔm"
    assert by_cell[("37", "f")]["Manual_Transcription"] == "mai̯kʰop | mai̯ragu"
    assert by_cell[("37", "f")]["Similarity_Groups"] == "1|2"
    assert by_cell[("37", "a")]["Manual_Transcription"] == "mikʰopʼ"
    assert by_cell[("37", "j")]["Manual_Transcription"] == "sorkʰao̯"
    assert by_cell[("38", "m")]["Manual_Transcription"] == "kʰan | alubʰuta"
    assert by_cell[("38", "k")]["Manual_Transcription"] == "pʰan | alu"
    assert by_cell[("38", "a")]["Manual_Transcription"] == "tʰa bultʃʰu"
    assert by_cell[("39", "0")]["Manual_Transcription"] == "pʰulkopi"
    assert by_cell[("40", "0")]["Manual_Transcription"] == "badʰakopi"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("36", "p"), ("39", "p"), ("40", "p")}


def test_ninth_manual_block_preserves_tree_branch_leaf_and_repeated_site():
    manual = load_module("garobd_manual_041_045", CHUNKS / "hand_keyed_items_041_045.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_041_045_cells.tsv")
    assert generated == rows
    assert len(lines) == 28
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(41, 46) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 84
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("41", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("41", "d")]["Manual_Transcription"] == "bai̯gon"
    assert by_cell[("41", "a")]["Manual_Transcription"] == "barɨŋ"
    assert by_cell[("41", "b")]["Manual_Transcription"] == "bantao̯"
    assert by_cell[("43", "p")]["Manual_Transcription"] == "ə dia̯ŋ"
    assert by_cell[("43", "0")]["Manual_Transcription"] == "gatʃʰ"
    assert by_cell[("44", "0")]["Manual_Transcription"] == "ɖal"
    assert by_cell[("44", "f")]["Manual_Transcription"] == "tʃɛkʃi"
    assert by_cell[("44", "j")]["Manual_Transcription"] == "rɨka"
    assert by_cell[("45", "b")]["Manual_Transcription"] == "lai̯tʃak | lai̯tʃak"
    assert by_cell[("45", "b")]["Similarity_Groups"] == "2|4"
    assert by_cell[("45", "a")]["Manual_Transcription"] == "bidʒakʼ"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("41", "p"), ("42", "p")}


def test_tenth_manual_block_preserves_root_repetitions_and_small_marks():
    manual = load_module("garobd_manual_046_050", CHUNKS / "hand_keyed_items_046_050.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_046_050_cells.tsv")
    assert generated == rows
    assert len(lines) == 36
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(46, 51) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 88
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("46", "0")]["Manual_Transcription"] == "kaʈa"
    assert by_cell[("46", "p")]["Manual_Transcription"] == "tʃiʔ"
    assert by_cell[("47", "a")]["Manual_Transcription"] == "ɖʒaʔdɨl"
    assert by_cell[("47", "f")]["Manual_Transcription"] == "tʃaʔdɨl | tʃaʔdɨl"
    assert by_cell[("47", "f")]["Similarity_Groups"] == "1|2"
    assert by_cell[("47", "l")]["Manual_Transcription"] == "tʃadɨl | tʃadɨl"
    assert by_cell[("47", "j")]["Manual_Transcription"] == "tʰɔtʼ"
    assert by_cell[("48", "m")]["Manual_Transcription"] == "wakai̯"
    assert by_cell[("48", "0")]["Manual_Transcription"] == "bãʃ"
    assert by_cell[("49", "n")]["Manual_Transcription"] == "bɛtʰei̯"
    assert by_cell[("49", "0")]["Manual_Transcription"] == "pʰɔl"
    assert by_cell[("50", "e")]["Manual_Transcription"] == "tʰai̯ʔbroŋ"
    assert by_cell[("50", "p")]["Manual_Transcription"] == "su ə rəm"
    assert by_cell[("50", "0")]["Manual_Transcription"] == "kaʈʰal"


def test_eleventh_manual_block_preserves_seed_repetitions_and_exact_blanks():
    manual = load_module("garobd_manual_051_055", CHUNKS / "hand_keyed_items_051_055.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_051_055_cells.tsv")
    assert generated == rows
    assert len(lines) == 48
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(51, 56) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 82,
        "source_blank": 3,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 90
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("51", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("51", "d")]["Manual_Transcription"] == "narikɛl"
    assert by_cell[("52", "a")]["Manual_Transcription"] == "tʰiʔrɨk"
    assert by_cell[("52", "p")]["Manual_Transcription"] == "kai̯tʼ"
    assert by_cell[("53", "e")]["Manual_Transcription"] == "tʰai̯ʔgatʃʰu"
    assert by_cell[("53", "i")]["Manual_Transcription"] == "tʰiʔgatʃu"
    assert by_cell[("53", "p")]["Manual_Transcription"] == "suʔpia̯ŋ"
    assert by_cell[("54", "l")]["Review_Status"] == "source_blank"
    assert by_cell[("54", "k")]["Manual_Transcription"] == "sɨntɨu̯"
    assert by_cell[("55", "b")]["Review_Status"] == "source_blank"
    assert by_cell[("55", "d")]["Manual_Transcription"] == "bigron | bigron"
    assert by_cell[("55", "d")]["Similarity_Groups"] == "2|3"
    assert by_cell[("55", "f")]["Manual_Transcription"] == "bigoron | bigoron | goron | goron"
    assert by_cell[("55", "f")]["Similarity_Groups"] == "2|4|4|A"
    assert by_cell[("55", "o")]["Manual_Transcription"] == "bɨtʃʰri | bɨtʃʰri"
    assert by_cell[("55", "j")]["Manual_Transcription"] == "tʃɨlɨi̯"
    assert by_cell[("55", "p")]["Manual_Transcription"] == "kʰut lia̯ŋ"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("51", "p"), ("54", "l"), ("55", "b")}


def test_twelfth_manual_block_preserves_sugarcane_betelnut_repetitions_and_blanks():
    manual = load_module("garobd_manual_056_060", CHUNKS / "hand_keyed_items_056_060.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_056_060_cells.tsv")
    assert generated == rows
    assert len(lines) == 38
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(56, 61) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 107
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("56", "l")]["Review_Status"] == "source_blank"
    assert by_cell[("56", "d")]["Manual_Transcription"] == "girɨtʼ | girɨtʼ"
    assert by_cell[("56", "d")]["Similarity_Groups"] == "1|2"
    assert by_cell[("56", "e")]["Manual_Transcription"] == "gorutʼ | gorutʼ"
    assert by_cell[("56", "j")]["Manual_Transcription"] == "kʰrui̯tʼ | kʰrui̯tʼ"
    assert by_cell[("56", "j")]["Similarity_Groups"] == "1|5"
    assert by_cell[("56", "m")]["Manual_Transcription"] == "golotʼ | golotʼ"
    assert by_cell[("56", "p")]["Manual_Transcription"] == "kʰlui̯t | kʰlui̯t"
    assert by_cell[("57", "b")]["Manual_Transcription"] == "goja | goja"
    assert by_cell[("57", "e")]["Manual_Transcription"] == "guwai̯ | guwai̯"
    assert by_cell[("57", "g")]["Manual_Transcription"] == "gui | gui"
    assert by_cell[("57", "a")]["Manual_Transcription"] == "guwa | guwa"
    assert by_cell[("58", "b")]["Manual_Transcription"] == "tʃunu"
    assert by_cell[("58", "0")]["Manual_Transcription"] == "tʃun"
    assert by_cell[("59", "l")]["Manual_Transcription"] == "tʃɨu̯"
    assert by_cell[("59", "p")]["Manual_Transcription"] == "kia̯t"
    assert by_cell[("59", "0")]["Manual_Transcription"] == "mɔd"
    assert by_cell[("60", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("60", "j")]["Manual_Transcription"] == "hɨmbu"
    assert by_cell[("60", "0")]["Manual_Transcription"] == "dudʰ"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("56", "l"), ("60", "p")}


def test_thirteenth_manual_block_preserves_oil_meat_salt_onion_garlic_marks_and_blanks():
    manual = load_module("garobd_manual_061_065", CHUNKS / "hand_keyed_items_061_065.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_061_065_cells.tsv")
    assert generated == rows
    assert len(lines) == 34
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(61, 66) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 83
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("61", "j")]["Manual_Transcription"] == "sonʲɛŋ"
    assert by_cell[("61", "p")]["Manual_Transcription"] == "suʔ nia̯ŋ"
    assert by_cell[("61", "0")]["Manual_Transcription"] == "tɛl"
    assert by_cell[("62", "b")]["Manual_Transcription"] == "pɛkʼɛn"
    assert by_cell[("62", "j")]["Manual_Transcription"] == "mɨn"
    assert by_cell[("62", "l")]["Manual_Transcription"] == "randai̯"
    assert by_cell[("62", "0")]["Manual_Transcription"] == "maŋʃo"
    assert by_cell[("63", "d")]["Manual_Transcription"] == "kʰai̯ʃum"
    assert by_cell[("63", "a")]["Manual_Transcription"] == "kʰasɨm"
    assert by_cell[("63", "j")]["Manual_Transcription"] == "mlɨk"
    assert by_cell[("63", "0")]["Manual_Transcription"] == "lɔbon / nun"
    assert by_cell[("64", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("64", "b")]["Manual_Transcription"] == "pia̯o̯"
    assert by_cell[("64", "l")]["Manual_Transcription"] == "rai̯sun"
    assert by_cell[("64", "n")]["Manual_Transcription"] == "gitʃakal rɔʃun"
    assert by_cell[("64", "0")]["Manual_Transcription"] == "pɛa̯dʒ"
    assert by_cell[("65", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("65", "a")]["Manual_Transcription"] == "nasɨn dukʼkʰi"
    assert by_cell[("65", "d")]["Manual_Transcription"] == "nasɨn gipʼpok"
    assert by_cell[("65", "l")]["Manual_Transcription"] == "rɔʃun"
    assert by_cell[("65", "k")]["Manual_Transcription"] == "ruʃun lɨʔ"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("64", "p"), ("65", "p")}


def test_fourteenth_manual_block_preserves_pepper_repetitions_affricate_ties_and_blanks():
    manual = load_module("garobd_manual_066_070", CHUNKS / "hand_keyed_items_066_070.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_066_070_cells.tsv")
    assert generated == rows
    assert len(lines) == 36
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(66, 71) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 87
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("66", "l")]["Manual_Transcription"] == "d͜ʒɨrɨk"
    assert by_cell[("66", "f")]["Manual_Transcription"] == "d͜ʒaʔlukʼ | d͜ʒaʔlukʼ"
    assert by_cell[("66", "f")]["Similarity_Groups"] == "1|2"
    assert by_cell[("66", "a")]["Manual_Transcription"] == "d͜ʒalɨkʼ | d͜ʒalɨkʼ"
    assert by_cell[("66", "e")]["Manual_Transcription"] == "d͜ʒaʔlukʰa"
    assert by_cell[("66", "i")]["Manual_Transcription"] == "d͜ʒalɨkʰa"
    assert by_cell[("66", "0")]["Manual_Transcription"] == "morɨtʃ"
    assert by_cell[("66", "p")]["Manual_Transcription"] == "suʔ sat sao̯"
    assert by_cell[("67", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("67", "o")]["Manual_Transcription"] == "moŋmau̯"
    assert by_cell[("67", "j")]["Manual_Transcription"] == "jao̯ba"
    assert by_cell[("68", "b")]["Manual_Transcription"] == "matʼsa"
    assert by_cell[("68", "a")]["Manual_Transcription"] == "matʼtʃʰa"
    assert by_cell[("68", "0")]["Manual_Transcription"] == "bagʰ"
    assert by_cell[("69", "i")]["Manual_Transcription"] == "makʼbɨl"
    assert by_cell[("69", "p")]["Manual_Transcription"] == "dɨŋ ŋem"
    assert by_cell[("69", "0")]["Manual_Transcription"] == "bʰaluk"
    assert by_cell[("70", "a")]["Review_Status"] == "source_blank"
    assert by_cell[("70", "b")]["Manual_Transcription"] == "matʃao̯"
    assert by_cell[("70", "i")]["Manual_Transcription"] == "matʼtʃʰok"
    assert by_cell[("70", "d")]["Manual_Transcription"] == "matʼtʃok"
    assert by_cell[("70", "p")]["Manual_Transcription"] == "skao̯"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("67", "p"), ("70", "a")}


def test_fifteenth_manual_block_preserves_snake_repetitions_combining_marks_and_blanks():
    manual = load_module("garobd_manual_071_075", CHUNKS / "hand_keyed_items_071_075.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_071_075_cells.tsv")
    assert generated == rows
    assert len(lines) == 56
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(71, 76) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 78,
        "source_blank": 7,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 91
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("71", "f")]["Review_Status"] == "source_blank"
    assert by_cell[("71", "b")]["Manual_Transcription"] == "kao̯ i"
    assert by_cell[("71", "p")]["Manual_Transcription"] == "tʃɨ riʔ"
    assert by_cell[("71", "j")]["Manual_Transcription"] == "tʃrɨʔ"
    assert by_cell[("72", "l")]["Review_Status"] == "source_blank"
    assert by_cell[("72", "b")]["Manual_Transcription"] == "hed͜ʒabari"
    assert by_cell[("72", "j")]["Manual_Transcription"] == "kʰorgoʃ | tʃahurɨn"
    assert by_cell[("72", "j")]["Similarity_Groups"] == "1|5"
    assert by_cell[("73", "i")]["Manual_Transcription"] == "tʃɨpʼpʰu | tʃʰɨpʼpʰu | tʃɨpʼpʰu"
    assert by_cell[("73", "i")]["Similarity_Groups"] == "1|1|3"
    assert by_cell[("73", "c")]["Manual_Transcription"] == "duɸu"
    assert by_cell[("73", "j")]["Manual_Transcription"] == "bsɨi̯n"
    assert by_cell[("73", "p")]["Manual_Transcription"] == "msei̯n"
    assert by_cell[("73", "l")]["Manual_Transcription"] == "dɨpɨu̯ | dɨpɨu̯"
    assert by_cell[("74", "a")]["Manual_Transcription"] == "arɨŋkʰa"
    assert by_cell[("74", "j")]["Manual_Transcription"] == "arɪŋga"
    assert by_cell[("74", "p")]["Manual_Transcription"] == "hariŋga"
    assert by_cell[("75", "l")]["Review_Status"] == "source_blank"
    assert by_cell[("75", "a")]["Manual_Transcription"] == "kʰantʃʰidɨk"
    assert by_cell[("75", "j")]["Manual_Transcription"] == "malɛŋkʰao̯"
    assert by_cell[("75", "n")]["Manual_Transcription"] == "hantɪkʼka"
    assert by_cell[("75", "0")]["Manual_Transcription"] == "tɪktɪkki"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("71", "f"), ("72", "l"), ("72", "o"), ("72", "p"), ("75", "l"), ("75", "o"), ("75", "p")}


def test_sixteenth_manual_block_preserves_turtle_frog_cat_cow_repetitions_and_marks():
    manual = load_module("garobd_manual_076_080", CHUNKS / "hand_keyed_items_076_080.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_076_080_cells.tsv")
    assert generated == rows
    assert len(lines) == 40
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(76, 81) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 97
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("76", "d")]["Manual_Transcription"] == "tʃapʼpa | tʃid͜ʒoŋ"
    assert by_cell[("76", "d")]["Similarity_Groups"] == "1|3"
    assert by_cell[("76", "k")]["Manual_Transcription"] == "katʰua̯ | dkar"
    assert by_cell[("76", "e")]["Manual_Transcription"] == "kʰatʼtʰua̯"
    assert by_cell[("76", "0")]["Manual_Transcription"] == "kɔttʃʰop"
    assert by_cell[("77", "m")]["Manual_Transcription"] == "luklak | bɛŋboŋ"
    assert by_cell[("77", "p")]["Manual_Transcription"] == "heruʔ"
    assert by_cell[("78", "l")]["Manual_Transcription"] == "kɨi"
    assert by_cell[("78", "p")]["Manual_Transcription"] == "kʰsu"
    assert by_cell[("79", "f")]["Manual_Transcription"] == "mɛŋgao̯"
    assert by_cell[("79", "d")]["Manual_Transcription"] == "mɛŋgou̯"
    assert by_cell[("79", "l")]["Manual_Transcription"] == "bɨi̯ra"
    assert by_cell[("79", "b")]["Manual_Transcription"] == "bilai̯"
    assert by_cell[("79", "p")]["Manual_Transcription"] == "mio̯"
    assert by_cell[("80", "k")]["Manual_Transcription"] == "mɨʔsɨu̯ | mɨʔsɨu̯"
    assert by_cell[("80", "j")]["Manual_Transcription"] == "maʔsɨu̯ | maʔsɨu̯"
    assert by_cell[("80", "b")]["Manual_Transcription"] == "maʔsu | maʔsu"
    assert by_cell[("80", "p")]["Manual_Transcription"] == "mɨsɨ"
    assert by_cell[("80", "0")]["Manual_Transcription"] == "goru"


def test_seventeenth_manual_block_preserves_buffalo_tail_goat_pig_marks_and_blank():
    manual = load_module("garobd_manual_081_085", CHUNKS / "hand_keyed_items_081_085.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_081_085_cells.tsv")
    assert generated == rows
    assert len(lines) == 34
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(81, 86) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 86
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("81", "f")]["Manual_Transcription"] == "moi̯ʃi"
    assert by_cell[("81", "e")]["Manual_Transcription"] == "muɨʃi"
    assert by_cell[("81", "l")]["Manual_Transcription"] == "tʃɨndɨk"
    assert by_cell[("81", "j")]["Manual_Transcription"] == "tʃɛrɨk"
    assert by_cell[("81", "0")]["Manual_Transcription"] == "mohiʃ"
    assert by_cell[("82", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("82", "a")]["Manual_Transcription"] == "groŋ"
    assert by_cell[("82", "j")]["Manual_Transcription"] == "rɨŋ"
    assert by_cell[("83", "l")]["Manual_Transcription"] == "diʔmi | diʔmi"
    assert by_cell[("83", "l")]["Similarity_Groups"] == "1|2"
    assert by_cell[("83", "e")]["Manual_Transcription"] == "kʰiʔmai̯"
    assert by_cell[("83", "0")]["Manual_Transcription"] == "lɛd͜ʒ"
    assert by_cell[("84", "o")]["Manual_Transcription"] == "doʔmok"
    assert by_cell[("84", "a")]["Manual_Transcription"] == "domokʼ"
    assert by_cell[("84", "h")]["Manual_Transcription"] == "brɨn"
    assert by_cell[("84", "0")]["Manual_Transcription"] == "tʃʰagol"
    assert by_cell[("85", "a")]["Manual_Transcription"] == "wakʼ"
    assert by_cell[("85", "j")]["Manual_Transcription"] == "snʲaŋ"
    assert by_cell[("85", "0")]["Manual_Transcription"] == "ʃukor"


def test_eighteenth_manual_block_preserves_rat_chicken_egg_fish_duck_marks_and_blank():
    manual = load_module("garobd_manual_086_090", CHUNKS / "hand_keyed_items_086_090.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_086_090_cells.tsv")
    assert generated == rows
    assert len(lines) == 39
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(86, 91) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 85
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("86", "b")]["Manual_Transcription"] == "moʃai̯"
    assert by_cell[("86", "n")]["Manual_Transcription"] == "moʃei̯"
    assert by_cell[("86", "g")]["Manual_Transcription"] == "mosei̯"
    assert by_cell[("86", "l")]["Manual_Transcription"] == "mid͜ʒutʼ"
    assert by_cell[("87", "p")]["Manual_Transcription"] == "sɨʔer"
    assert by_cell[("87", "k")]["Manual_Transcription"] == "siʔɛr"
    assert by_cell[("87", "e")]["Manual_Transcription"] == "dau̯"
    assert by_cell[("88", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("88", "a")]["Manual_Transcription"] == "bɨtʼtʃi"
    assert by_cell[("88", "b")]["Manual_Transcription"] == "pitɪk"
    assert by_cell[("88", "f")]["Manual_Transcription"] == "dao̯tʃʰi"
    assert by_cell[("88", "l")]["Manual_Transcription"] == "tɨi̯"
    assert by_cell[("88", "0")]["Manual_Transcription"] == "d̪im"
    assert by_cell[("89", "a")]["Manual_Transcription"] == "naʔtʰokʼ"
    assert by_cell[("89", "0")]["Manual_Transcription"] == "matʃʰ"
    assert by_cell[("90", "o")]["Manual_Transcription"] == "gagakʼ | dogɛpʼ"
    assert by_cell[("90", "o")]["Similarity_Groups"] == "1|4"
    assert by_cell[("90", "j")]["Manual_Transcription"] == "dao̯gɛpʼ"
    assert by_cell[("90", "p")]["Manual_Transcription"] == "dao̯gep"
    assert by_cell[("90", "0")]["Manual_Transcription"] == "haʃ"


def test_nineteenth_manual_block_preserves_bird_insect_cockroach_bee_fly_marks_and_blank():
    manual = load_module("garobd_manual_091_095", CHUNKS / "hand_keyed_items_091_095.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_091_095_cells.tsv")
    assert generated == rows
    assert len(lines) == 44
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(91, 96) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 89
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("91", "o")]["Review_Status"] == "source_blank"
    assert by_cell[("91", "b")]["Manual_Transcription"] == "tau̯"
    assert by_cell[("91", "p")]["Manual_Transcription"] == "sɪm"
    assert by_cell[("91", "0")]["Manual_Transcription"] == "pakʰi"
    assert by_cell[("92", "a")]["Manual_Transcription"] == "d͜ʒoŋ"
    assert by_cell[("92", "e")]["Manual_Transcription"] == "d͜ʒoŋʔʃu"
    assert by_cell[("92", "j")]["Manual_Transcription"] == "kʰnia̯ŋ"
    assert by_cell[("93", "n")]["Manual_Transcription"] == "sɛʔlou̯ | sɛʔlou̯"
    assert by_cell[("93", "n")]["Similarity_Groups"] == "1|2"
    assert by_cell[("93", "m")]["Manual_Transcription"] == "saluŋ | tɛlapoka"
    assert by_cell[("93", "a")]["Manual_Transcription"] == "ʃɛʔlukʼ"
    assert by_cell[("94", "c")]["Manual_Transcription"] == "nija (tʃoŋ)"
    assert by_cell[("94", "a")]["Manual_Transcription"] == "bid͜ʒa"
    assert by_cell[("94", "0")]["Manual_Transcription"] == "mou̯matʃʰi"
    assert by_cell[("95", "g")]["Manual_Transcription"] == "kʰampʰi | kʰampʰi"
    assert by_cell[("95", "g")]["Similarity_Groups"] == "1|3"
    assert by_cell[("95", "e")]["Manual_Transcription"] == "kʰanʔpʰi"
    assert by_cell[("95", "j")]["Manual_Transcription"] == "pʰɛŋrai̯"


def test_twentieth_manual_block_preserves_spider_ant_mosquito_head_face_marks_and_blanks():
    manual = load_module("garobd_manual_096_100", CHUNKS / "hand_keyed_items_096_100.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_096_100_cells.tsv")
    assert generated == rows
    assert len(lines) == 45
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(96, 101) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 88
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("96", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("96", "b")]["Manual_Transcription"] == "abrɛkʼ"
    assert by_cell[("96", "j")]["Manual_Transcription"] == "nʲam"
    assert by_cell[("96", "k")]["Manual_Transcription"] == "pokɨda"
    assert by_cell[("97", "d")]["Manual_Transcription"] == "ʃimal | ʃimal"
    assert by_cell[("97", "d")]["Similarity_Groups"] == "2|3"
    assert by_cell[("97", "l")]["Manual_Transcription"] == "samal | samal"
    assert by_cell[("97", "g")]["Manual_Transcription"] == "katʃɨŋ"
    assert by_cell[("97", "p")]["Manual_Transcription"] == "dɨkʰɨ"
    assert by_cell[("98", "p")]["Manual_Transcription"] == "d͜ʒɨkai̯ŋ"
    assert by_cell[("98", "j")]["Manual_Transcription"] == "tʃɨkai̯n"
    assert by_cell[("99", "n")]["Manual_Transcription"] == "ʃɛkʰou̯ | ʃɛkʰou̯"
    assert by_cell[("99", "n")]["Similarity_Groups"] == "3|4"
    assert by_cell[("99", "j")]["Manual_Transcription"] == "kʰlɪʔ"
    assert by_cell[("100", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("100", "a")]["Manual_Transcription"] == "mɨkʼkʰaŋ"
    assert by_cell[("100", "l")]["Manual_Transcription"] == "mɨkʰɨŋ"
    assert by_cell[("100", "e")]["Manual_Transcription"] == "mukʼkʰaŋ"


def test_twenty_first_manual_block_preserves_neck_hair_eye_nose_ear_marks_and_repeats():
    manual = load_module("garobd_manual_101_105", CHUNKS / "hand_keyed_items_101_105.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_101_105_cells.tsv")
    assert generated == rows
    assert len(lines) == 38
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(101, 106) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 91
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("101", "o")]["Manual_Transcription"] == "gɨtʼdok"
    assert by_cell[("101", "l")]["Manual_Transcription"] == "tokɾɛŋ"
    assert by_cell[("102", "a")]["Manual_Transcription"] == "kʰɨnni"
    assert by_cell[("102", "b")]["Manual_Transcription"] == "kʰau̯"
    assert by_cell[("102", "j")]["Manual_Transcription"] == "snʲɨk"
    assert by_cell[("103", "l")]["Manual_Transcription"] == "mɨkɾɛŋ | mɨkɾɛŋ"
    assert by_cell[("103", "l")]["Similarity_Groups"] == "1|2"
    assert by_cell[("103", "g")]["Manual_Transcription"] == "mɨkɾon | mɨkɾon"
    assert by_cell[("103", "d")]["Manual_Transcription"] == "mukɾoŋ | mukɾoŋ"
    assert by_cell[("103", "b")]["Manual_Transcription"] == "mokʼkon"
    assert by_cell[("104", "j")]["Manual_Transcription"] == "lɨmʊt"
    assert by_cell[("104", "p")]["Manual_Transcription"] == "lɨmut"
    assert by_cell[("104", "b")]["Manual_Transcription"] == "nakʼkuŋ"
    assert by_cell[("105", "a")]["Manual_Transcription"] == "natʃɨl"
    assert by_cell[("105", "b")]["Manual_Transcription"] == "nakʰar"


def test_twenty_second_manual_block_preserves_cheek_chin_mouth_tongue_tooth_marks_and_blank():
    manual = load_module("garobd_manual_106_110", CHUNKS / "hand_keyed_items_106_110.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_106_110_cells.tsv")
    assert generated == rows
    assert len(lines) == 48
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(106, 111) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 99
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("106", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("106", "e")]["Manual_Transcription"] == "pʰai̯tʰopʼ | pʰai̯tʰopʼ | pʰai̯tʰopʼ"
    assert by_cell[("106", "e")]["Similarity_Groups"] == "1|2|3"
    assert by_cell[("106", "a")]["Manual_Transcription"] == "pʰitʰopʼ | pʰitʰopʼ"
    assert by_cell[("106", "n")]["Manual_Transcription"] == "pʰɛtʰɪŋ"
    assert by_cell[("106", "h")]["Manual_Transcription"] == "pʰitʰɨŋ"
    assert by_cell[("107", "a")]["Manual_Transcription"] == "kʰudumbok | kʰudumbok"
    assert by_cell[("107", "e")]["Manual_Transcription"] == "kʰuʔdubu"
    assert by_cell[("108", "l")]["Manual_Transcription"] == "kʰutʃukʼ | kʰutʃukʼ"
    assert by_cell[("109", "e")]["Manual_Transcription"] == "ʃɛlɛbakʼ | ʃɛlɛbakʼ"
    assert by_cell[("109", "e")]["Similarity_Groups"] == "3|4"
    assert by_cell[("109", "j")]["Manual_Transcription"] == "tʰɨloi̯tʼ"
    assert by_cell[("109", "0")]["Manual_Transcription"] == "dʒɪb"
    assert by_cell[("110", "j")]["Manual_Transcription"] == "moi̯n"
    assert by_cell[("110", "0")]["Manual_Transcription"] == "dãt"


def test_twenty_third_manual_block_preserves_elbow_hand_palm_finger_fingernail_marks_and_blank():
    manual = load_module("garobd_manual_111_115", CHUNKS / "hand_keyed_items_111_115.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_111_115_cells.tsv")
    assert generated == rows
    assert len(lines) == 44
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(111, 116) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 91
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("111", "p")]["Review_Status"] == "source_blank"
    assert by_cell[("111", "o")]["Manual_Transcription"] == "d͜ʒakʼʃkʰu"
    assert by_cell[("111", "c")]["Manual_Transcription"] == "tʃa gilai̯"
    assert by_cell[("111", "d")]["Manual_Transcription"] == "d͜ʒakʼtʃukʼ"
    assert by_cell[("112", "a")]["Manual_Transcription"] == "d͜ʒakʼ"
    assert by_cell[("112", "b")]["Manual_Transcription"] == "t͜ʒakʼaprakʼ"
    assert by_cell[("112", "j")]["Manual_Transcription"] == "kʰtɨi̯"
    assert by_cell[("113", "a")]["Manual_Transcription"] == "d͜ʒakʼpʰa | d͜ʒakʼpʰa"
    assert by_cell[("113", "a")]["Similarity_Groups"] == "1|2"
    assert by_cell[("113", "f")]["Manual_Transcription"] == "d͜ʒakʼpʰatʰai̯"
    assert by_cell[("113", "p")]["Manual_Transcription"] == "sla kʰtɪ"
    assert by_cell[("114", "j")]["Manual_Transcription"] == "lutɨi̯"
    assert by_cell[("115", "a")]["Manual_Transcription"] == "d͜ʒakʼsɨkʰɨl"
    assert by_cell[("115", "e")]["Manual_Transcription"] == "d͜ʒakʼʃukʰul"
    assert by_cell[("115", "m")]["Manual_Transcription"] == "tʃakʼʃikʰor"


def test_twenty_fourth_manual_block_preserves_knee_foot_bone_fat_skin_marks_repeats_and_blanks():
    manual = load_module("garobd_manual_116_120", CHUNKS / "hand_keyed_items_116_120.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_116_120_cells.tsv")
    assert generated == rows
    assert len(lines) == 45
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(116, 121) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 81,
        "source_blank": 4,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 85
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("116", "a")]["Manual_Transcription"] == "d͜ʒaʃukʰu | d͜ʒaʃukʰu"
    assert by_cell[("116", "a")]["Similarity_Groups"] == "1|2"
    assert by_cell[("116", "d")]["Manual_Transcription"] == "d͜ʒaʃuku | d͜ʒaʃuku"
    assert by_cell[("116", "e")]["Manual_Transcription"] == "d͜ʒaʔʃukʰu"
    assert by_cell[("116", "j")]["Manual_Transcription"] == "eŋmao̯ɾɛŋ"
    assert by_cell[("116", "p")]["Manual_Transcription"] == "eŋmao̯ria̯ŋ"
    assert by_cell[("117", "b")]["Manual_Transcription"] == "tʃakɾɛŋ aprakʼ"
    assert by_cell[("117", "p")]["Manual_Transcription"] == "kid͜ʒia̯t"
    assert by_cell[("117", "j")]["Manual_Transcription"] == "kʰd͜ʒɛtʼ"
    assert by_cell[("118", "e")]["Manual_Transcription"] == "gɛɾɛŋ"
    assert by_cell[("118", "p")]["Manual_Transcription"] == "tʃia̯ŋ"
    assert by_cell[("118", "j")]["Manual_Transcription"] == "tʃiɛ̯ŋ"
    assert by_cell[("119", "d")]["Manual_Transcription"] == "mɨtʼdɨm"
    assert by_cell[("119", "j")]["Manual_Transcription"] == "kʰlɨi̯ŋ"
    assert by_cell[("119", "p")]["Manual_Transcription"] == "kʰlɪŋ"
    assert by_cell[("120", "a")]["Manual_Transcription"] == "bigɨl"
    assert by_cell[("120", "j")]["Manual_Transcription"] == "snɪʔ"
    assert by_cell[("120", "0")]["Manual_Transcription"] == "tʃamɾa"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("119", "a"), ("119", "b"), ("119", "i"), ("119", "l")}


def test_twenty_fifth_manual_block_preserves_blood_sweat_belly_heart_back_marks_repeats_and_blanks():
    manual = load_module("garobd_manual_121_125", CHUNKS / "hand_keyed_items_121_125.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_121_125_cells.tsv")
    assert generated == rows
    assert len(lines) == 42
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(121, 126) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 81,
        "source_blank": 4,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 83
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("121", "e")]["Manual_Transcription"] == "hanʔtʃʰi"
    assert by_cell[("121", "a")]["Manual_Transcription"] == "hantʃʰi"
    assert by_cell[("121", "l")]["Manual_Transcription"] == "tʰɨi̯"
    assert by_cell[("121", "p")]["Manual_Transcription"] == "sɨŋŋam"
    assert by_cell[("122", "g")]["Manual_Transcription"] == "ruʔutʃia"
    assert by_cell[("122", "n")]["Manual_Transcription"] == "rutʃia̯"
    assert by_cell[("122", "j")]["Manual_Transcription"] == "d͜ʒɨlupʼ"
    assert by_cell[("122", "m")]["Manual_Transcription"] == "tuŋgoa̯"
    assert by_cell[("122", "0")]["Manual_Transcription"] == "gʰam"
    assert by_cell[("123", "m")]["Manual_Transcription"] == "pipukʼ | pipukʼ"
    assert by_cell[("123", "m")]["Similarity_Groups"] == "2|7"
    assert by_cell[("123", "j")]["Manual_Transcription"] == "kʰlao̯"
    assert by_cell[("123", "p")]["Manual_Transcription"] == "lau̯baʔ"
    assert by_cell[("123", "l")]["Manual_Transcription"] == "pipʰuʔ"
    assert by_cell[("124", "l")]["Manual_Transcription"] == "pikʰa | d͜ʒaŋgi"
    assert by_cell[("124", "l")]["Similarity_Groups"] == "1|3"
    assert by_cell[("124", "0")]["Manual_Transcription"] == "ridɔi̯"
    assert by_cell[("125", "a")]["Manual_Transcription"] == "d͜ʒaŋgɨl"
    assert by_cell[("125", "p")]["Manual_Transcription"] == "pʰat"
    assert by_cell[("125", "j")]["Manual_Transcription"] == "pʰatʼ"
    assert by_cell[("125", "0")]["Manual_Transcription"] == "pitʰ"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("122", "b"), ("122", "e"), ("124", "b"), ("124", "p")}


def test_twenty_sixth_manual_block_preserves_body_person_man_woman_father_marks_repeat_and_blank():
    manual = load_module("garobd_manual_126_130", CHUNKS / "hand_keyed_items_126_130.py")
    lines = manual.line_rows()
    generated = manual.cell_rows(lines)
    rows = read_tsv(CHUNKS / "items_126_130_cells.tsv")
    assert generated == rows
    assert len(lines) == 40
    assert len(rows) == 5 * 17 == 85
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(126, 131) for code in manual.SITE_CODES
    }
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 85
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    assert all(not any(key.casefold().startswith(("ocr", "legacy")) for key in row) for row in lines + rows)

    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("126", "h")]["Review_Status"] == "source_blank"
    assert by_cell[("126", "d")]["Manual_Transcription"] == "bimaŋ | bɛʔɛn"
    assert by_cell[("126", "d")]["Similarity_Groups"] == "1|3"
    assert by_cell[("126", "p")]["Manual_Transcription"] == "mɨm pʰat brɨ"
    assert by_cell[("126", "l")]["Manual_Transcription"] == "randai̯"
    assert by_cell[("127", "e")]["Manual_Transcription"] == "mandai̯"
    assert by_cell[("127", "h")]["Manual_Transcription"] == "mandei̯"
    assert by_cell[("127", "k")]["Manual_Transcription"] == "brɨu̯"
    assert by_cell[("128", "a")]["Manual_Transcription"] == "mɛʔɛʃa"
    assert by_cell[("128", "d")]["Manual_Transcription"] == "mɛʔaʃa"
    assert by_cell[("128", "j")]["Manual_Transcription"] == "kʰoŋkoraŋ"
    assert by_cell[("129", "a")]["Manual_Transcription"] == "mitʃɨkʼʃa"
    assert by_cell[("129", "e")]["Manual_Transcription"] == "mitʃikʼʃa"
    assert by_cell[("129", "p")]["Manual_Transcription"] == "rao̯kmao̯"
    assert by_cell[("129", "j")]["Manual_Transcription"] == "rokʼmao̯"
    assert by_cell[("129", "l")]["Manual_Transcription"] == "gɨwuɨ̯"
    assert by_cell[("129", "0")]["Manual_Transcription"] == "mohɨla"
    assert by_cell[("130", "e")]["Manual_Transcription"] == "apʼpʰa"
    assert by_cell[("130", "0")]["Manual_Transcription"] == "baba"
    assert by_cell[("130", "n")]["Manual_Transcription"] == "babu"


def test_twenty_seventh_manual_block_preserves_mother_husband_wife_son_daughter_marks_and_repeat():
    manual = load_module("garobd_manual_131_135", CHUNKS / "hand_keyed_items_131_135.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_131_135_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 49
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 86
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("131", "p")]["Manual_Transcription"] == "bei̯ | bei̯"
    assert by_cell[("131", "p")]["Similarity_Groups"] == "2|4"
    assert by_cell[("132", "l")]["Manual_Transcription"] == "d͜ʒɨkʼbipʰa"
    assert by_cell[("133", "l")]["Manual_Transcription"] == "d͜ʒɨkʼgɨwuɨ̯"
    assert by_cell[("134", "a")]["Manual_Transcription"] == "mɛʔɛʃa pʰiʃa"
    assert by_cell[("134", "0")]["Manual_Transcription"] == "tʃʰɛlɛ"
    assert by_cell[("135", "d")]["Manual_Transcription"] == "mitʃɨkʼʃa dɛ"
    assert by_cell[("135", "g")]["Manual_Transcription"] == "mitʃɨkʼsa dei̯"
    assert by_cell[("135", "f")]["Manual_Transcription"] == "mitʃʰikʼ doi̯"
    assert by_cell[("135", "e")]["Manual_Transcription"] == "mitʃʰikʼʃa dei̯"
    assert by_cell[("135", "p")]["Manual_Transcription"] == "rao̯k mao̯"


def test_twenty_eighth_manual_block_preserves_sibling_friend_marks_and_overlaps():
    manual = load_module("garobd_manual_136_140", CHUNKS / "hand_keyed_items_136_140.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_136_140_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 41
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {"attested": 85}
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 89
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("136", "c")]["Manual_Transcription"] == "dada | kaka"
    assert by_cell[("136", "c")]["Similarity_Groups"] == "1|2"
    assert by_cell[("136", "m")]["Manual_Transcription"] == "pʰao̯ tʃuŋguwa"
    assert by_cell[("137", "c")]["Manual_Transcription"] == "ad͜ʒa | bai̯"
    assert by_cell[("137", "0")]["Manual_Transcription"] == "bɔro bon / didi"
    assert by_cell[("137", "p")]["Manual_Transcription"] == "hɨn min rao̯k mao̯"
    assert by_cell[("138", "e")]["Manual_Transcription"] == "d͜ʒoŋ | d͜ʒod͜ʒoŋ"
    assert by_cell[("138", "j")]["Manual_Transcription"] == "hɨmbu dodɨpʼ"
    assert by_cell[("138", "p")]["Manual_Transcription"] == "hɨnbu dudit"
    assert by_cell[("139", "p")]["Manual_Transcription"] == "hɨnbu rao̯kmao̯"
    assert by_cell[("140", "g")]["Manual_Transcription"] == "bad͜ʒu | bei̯ʃa"
    assert by_cell[("140", "g")]["Similarity_Groups"] == "1|3"
    assert by_cell[("140", "j")]["Manual_Transcription"] == "marlokʼ"


def test_twenty_ninth_manual_block_preserves_name_building_marks_repeats_and_blank():
    manual = load_module("garobd_manual_141_145", CHUNKS / "hand_keyed_items_141_145.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_141_145_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 34
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 84,
        "source_blank": 1,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 87
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("141", "a")]["Manual_Transcription"] == "bimɨŋ"
    assert by_cell[("141", "l")]["Manual_Transcription"] == "bimʊŋ"
    assert by_cell[("141", "d")]["Manual_Transcription"] == "bimuŋ"
    assert by_cell[("141", "p")]["Manual_Transcription"] == "kɨr tɨŋ"
    assert by_cell[("142", "p")]["Manual_Transcription"] == "tʃɨnoŋ"
    assert by_cell[("143", "j")]["Manual_Transcription"] == "jii̯n"
    assert by_cell[("143", "0")]["Manual_Transcription"] == "bari / gʰor"
    assert by_cell[("144", "e")]["Manual_Transcription"] == "doʔgatʃul | doʔgatʃul"
    assert by_cell[("144", "e")]["Similarity_Groups"] == "1|7"
    assert by_cell[("144", "i")]["Manual_Transcription"] == "doʔga | doʔga | doʔga"
    assert by_cell[("144", "i")]["Similarity_Groups"] == "2|3|7"
    assert by_cell[("144", "0")]["Manual_Transcription"] == "dɔrd͜ʒa"
    assert by_cell[("145", "c")]["Manual_Transcription"] == "kʰokai̯ dua̯r"
    assert by_cell[("145", "0")]["Manual_Transcription"] == "d͜ʒanala"
    blank = by_cell[("145", "p")]
    assert blank["Review_Status"] == "source_blank"
    assert blank["Source_Qualification"] == 'printed "no entry"'


def test_thirtieth_manual_block_preserves_roof_furnishing_ring_marks_repeats_and_blanks():
    manual = load_module("garobd_manual_146_150", CHUNKS / "hand_keyed_items_146_150.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_146_150_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 33
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 83,
        "source_blank": 2,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 87
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("146", "a")]["Manual_Transcription"] == "nokʼkʰɨŋ"
    assert by_cell[("146", "l")]["Manual_Transcription"] == "nukʰuŋ | nukʰuŋ"
    assert by_cell[("146", "l")]["Similarity_Groups"] == "1|5"
    assert by_cell[("146", "c")]["Manual_Transcription"] == "nukʰuraŋ | nukʰuraŋ"
    assert by_cell[("146", "0")]["Manual_Transcription"] == "tʃʰad / tʃal"
    assert by_cell[("147", "p")]["Manual_Transcription"] == "kɨn ruʔ"
    assert by_cell[("148", "0")]["Manual_Transcription"] == "balɪʃ"
    assert by_cell[("148", "p")]["Manual_Transcription"] == "kʰonkʰlɪʔ"
    assert by_cell[("149", "0")]["Manual_Transcription"] == "kɔmbol"
    assert by_cell[("150", "f")]["Manual_Transcription"] == "aŋdi"
    assert by_cell[("150", "j")]["Manual_Transcription"] == "sulutei̯"
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("146", "p"), ("149", "p")}


def test_thirty_first_manual_block_preserves_clothing_medicine_paper_needle_and_not_used_scope():
    manual = load_module("garobd_manual_151_155", CHUNKS / "hand_keyed_items_151_155.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_151_155_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 31
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 68,
        "not_used": 17,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 72
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("151", "g")]["Manual_Transcription"] == "tʃʰɨnna"
    assert by_cell[("151", "j")]["Manual_Transcription"] == "d͜ʒai̯n"
    assert by_cell[("151", "p")]["Manual_Transcription"] == "d͜ʒai̯n pʰoŋ"
    assert by_cell[("153", "b")]["Manual_Transcription"] == "pantʃakʼ"
    assert by_cell[("153", "p")]["Manual_Transcription"] == "duwai̯"
    assert by_cell[("154", "a")]["Manual_Transcription"] == "lɛkʼkʰa"
    assert by_cell[("154", "0")]["Manual_Transcription"] == "kagod͜ʒ"
    assert by_cell[("155", "g")]["Manual_Transcription"] == "ʃutʃʰi | ʃutʃʰi"
    assert by_cell[("155", "g")]["Similarity_Groups"] == "2|5"
    assert by_cell[("155", "d")]["Manual_Transcription"] == "ʃutʃi | ʃutʃi"
    assert by_cell[("155", "p")]["Manual_Transcription"] == "tʰɨr ria"
    not_used = [row for row in rows if row["Review_Status"] == "not_used"]
    assert {(row["Item"], row["Site_Code"]) for row in not_used} == {
        ("152", code) for code in manual.SITE_CODES
    }
    assert all(row["Source_Qualification"] == 'printed "[not used]" for whole item' for row in not_used)


def test_thirty_second_manual_block_preserves_thread_broom_spoon_hammer_overlaps_and_not_used_scope():
    manual = load_module("garobd_manual_156_160", CHUNKS / "hand_keyed_items_156_160.py")
    lines = manual.line_rows()
    rows = read_tsv(CHUNKS / "items_156_160_cells.tsv")
    assert manual.cell_rows(lines) == rows
    assert len(lines) == 30
    assert len(rows) == 85
    assert Counter(row["Review_Status"] for row in rows) == {
        "attested": 68,
        "not_used": 17,
    }
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Manual_Transcription"]) == 80
    assert all(not row["Uncertainty"] for row in rows)
    assert all(row["Reviewer_Declaration"] == manual.DECLARATION for row in lines + rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in lines + rows for value in row.values())
    by_cell = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_cell[("156", "a")]["Manual_Transcription"] == "kʰɨldɨŋ"
    assert by_cell[("156", "j")]["Manual_Transcription"] == "kʃai̯"
    assert by_cell[("156", "p")]["Manual_Transcription"] == "kʰsai̯"
    assert by_cell[("157", "0")]["Manual_Transcription"] == "d͜ʒʰaɾu"
    assert by_cell[("157", "j")]["Manual_Transcription"] == "tʃipʼnatʼ"
    assert by_cell[("158", "n")]["Manual_Transcription"] == "ata | tʃamotʃ"
    assert by_cell[("158", "n")]["Similarity_Groups"] == "2|3"
    assert by_cell[("160", "a")]["Manual_Transcription"] == "hatur | hatur"
    assert by_cell[("160", "a")]["Similarity_Groups"] == "1|4"
    assert by_cell[("160", "k")]["Manual_Transcription"] == "d͜ʒoŋ mnoʔ"
    assert by_cell[("160", "p")]["Manual_Transcription"] == "tɨrnim"
    not_used = [row for row in rows if row["Review_Status"] == "not_used"]
    assert {(row["Item"], row["Site_Code"]) for row in not_used} == {
        ("159", code) for code in manual.SITE_CODES
    }
    assert all(row["Source_Qualification"] == 'printed "[not used]" for whole item' for row in not_used)


def test_frozen_hashes_manifest_and_post_freeze_reconciliation_are_exact():
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    checkpoint = manifest["manual_review_checkpoint"]
    assert manifest["wordlist_physical_pages"] == [52, 93]
    assert manifest["wordlist_printed_pages"] == [45, 86]
    assert checkpoint["completed_items"] == [1, 160]
    assert checkpoint["pending_items"] == [161, 307]
    assert checkpoint["printed_response_lines"] == 1251
    assert checkpoint["conceptual_cells"] == 2720
    assert checkpoint["attested_cells"] == 2624
    assert checkpoint["source_conflict_cells"] == 1
    assert checkpoint["cells_with_attestations"] == 2625
    assert checkpoint["source_blank_cells"] == 61
    assert checkpoint["printed_no_entry_coordinates"] == 62
    assert checkpoint["not_used_cells"] == 34
    assert checkpoint["expanded_attested_responses"] == 2808
    assert checkpoint["unresolved_coordinates"] == [{
        "item": 12,
        "site_code": "p",
        "physical_page": 53,
        "printed_page": 46,
        "reason": "source prints both group-0 no entry and group-6 dɔm",
    }]

    chunks = checkpoint["chunks"]
    assert [chunk["items"] for chunk in chunks] == [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30], [31, 35], [36, 40], [41, 45], [46, 50], [51, 55], [56, 60], [61, 65], [66, 70], [71, 75], [76, 80], [81, 85], [86, 90], [91, 95], [96, 100], [101, 105], [106, 110], [111, 115], [116, 120], [121, 125], [126, 130], [131, 135], [136, 140], [141, 145], [146, 150], [151, 155], [156, 160]]
    for chunk in chunks:
        assert digest(HERE / chunk["generator"]["path"]) == chunk["generator"]["sha256"]
        assert digest(HERE / chunk["frozen_line_ledger"]["path"]) == chunk["frozen_line_ledger"]["sha256"]
        assert digest(HERE / chunk["frozen_cell_ledger"]["path"]) == chunk["frozen_cell_ledger"]["sha256"]
        reconciliation = chunk["post_freeze_reconciliation"]
        assert digest(HERE / reconciliation["path"]) == reconciliation["sha256"]

    rows = [
        *read_tsv(CHUNKS / "items_001_005_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_006_010_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_011_015_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_016_020_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_021_025_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_026_030_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_031_035_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_036_040_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_041_045_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_046_050_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_051_055_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_056_060_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_061_065_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_066_070_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_071_075_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_076_080_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_081_085_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_086_090_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_091_095_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_096_100_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_101_105_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_106_110_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_111_115_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_116_120_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_121_125_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_126_130_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_131_135_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_136_140_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_141_145_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_146_150_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_151_155_reconciliation.tsv"),
        *read_tsv(CHUNKS / "items_156_160_reconciliation.tsv"),
    ]
    assert len(rows) == 2872
    assert Counter(row["Legacy_Status"] for row in rows) == {"installed": 2405, "excluded": 466, "missing": 1}
    assert Counter(row["Exact_Codepoint_Equal"] for row in rows) == {"no": 1905, "yes": 967}
    assert Counter(row["Reconciliation_Disposition"] for row in rows) == {
        "legacy exact comparison match (audit-only; not verification)": 967,
        "legacy differs at codepoint level (audit-only; manual unchanged)": 1438,
        "manually recovered formerly excluded glyph sequence": 402,
        "manual source blank; legacy audit also records a printed gap": 62,
        "manual whole-item not-used; legacy audit also records the printed disposition": 2,
        "manual source record absent from legacy audit": 1,
    }
    assert all("did not supply, alter, or verify" in row["Independence_Note"] for row in rows)
