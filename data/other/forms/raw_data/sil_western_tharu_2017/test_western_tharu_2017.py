import csv
import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "western_tharu_import", ROOT / "import_western_tharu_2017.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MOD)


def test_canonical_source_and_exact_topology():
    assert MOD.PDF.exists()
    assert MOD.sha256(MOD.PDF) == MOD.PDF_SHA256
    assert (MOD.EXPECTED_ITEMS, MOD.EXPECTED_LISTS, MOD.EXPECTED_CELLS) == (210, 16, 3360)
    registry = MOD.load_registry()
    assert len(registry) == 16
    assert len(MOD.TARGETS) == 15
    assert registry["HIN"]["Scope"] == "control"
    assert registry["RNS_Sisaikhara"]["Code_Occurrence"] == "1"
    assert registry["RNS_Sisana"]["Code_Occurrence"] == "2"
    assert registry["RKM"]["Metadata_Code"] == "RKM"
    assert registry["RKM"]["Response_Code"] == "RkM"


def test_items_1_210_manual_ledger_is_complete_ocr_blind_nfc_and_resolved():
    rows = MOD.load_cells()
    assert len(rows) == 3360
    assert {(int(row["Item"]), row["Site_Key"]) for row in rows} == {
        (item, site) for item in range(1, 211) for site in MOD.load_registry()
    }
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=3261, source_blank=99
    )
    assert not any("ocr" in field.casefold() for field in rows[0])
    assert all(row["Reviewer_Declaration"] == MOD.DECLARATION for row in rows)
    assert all(row["Reviewer_Method"] == MOD.METHOD for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    assert MOD.summarize(rows) == {
        "reviewed_cells": 3360,
        "attested_cells": 3261,
        "source_blank_cells": 99,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "target_reviewed_cells": 3150,
        "control_reviewed_cells": 210,
        "target_candidate_forms": 3560,
        "control_candidate_forms": 290,
        "pending_cells": 0,
    }


def test_exact_blanks_alternatives_qualifier_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    assert [
        (item, site, row["PDF_Page"], row["Printed_Page"], row["Column"])
        for (item, site), row in rows.items() if row["Review_Status"] == "source_blank"
    ] == [
        (1, "CCC", "31", "26", "left"),
        (4, "CCC", "31 / 32", "26 / 27", "right / left"),
        (11, "RKM", "33", "28", "left"),
        (21, "DKS", "35", "30", "left"),
        (33, "CCC", "37", "32", "left / right"),
        (50, "CCC", "40", "35", "right"),
        (51, "BNT", "40", "35", "right"),
        (53, "BNT", "41", "36", "left"),
        (58, "CCC", "41 / 42", "36 / 37", "right / left"),
        (59, "CCC", "42", "37", "left"),
        (66, "DDK", "43", "38", "left"),
        (69, "CCC", "43", "38", "right"),
        (70, "RNS_Sisana", "43 / 44", "38 / 39", "right / left"),
        (71, "CCC", "44", "39", "left"),
        (73, "CCC", "44", "39", "left / right"),
        (76, "CCC", "45", "40", "left"),
        (79, "CCC", "45", "40", "right"),
        (81, "CCC", "46", "41", "left"),
        (84, "TkN", "46", "41", "right"),
        (85, "CCC", "46 / 47", "41 / 42", "right / left"),
        (87, "CCC", "47", "42", "left"),
        (104, "CCC", "50", "45", "left"),
        (105, "CCC", "50", "45", "left / right"),
        (106, "CCC", "50", "45", "right"),
        (107, "CCC", "50 / 51", "45 / 46", "right / left"),
        (109, "CCC", "51", "46", "left"),
        (110, "CCC", "51", "46", "left / right"),
        (111, "CCC", "51", "46", "right"),
        (112, "CCC", "51", "46", "right"),
        (113, "CCC", "51 / 52", "46 / 47", "right / left"),
        (114, "CCC", "52", "47", "left"),
        (115, "CCC", "52", "47", "left"),
        (116, "CCC", "52", "47", "left / right"),
        (120, "CCC", "53", "48", "left"),
        (125, "BNT", "54", "49", "left"),
        (125, "RNS_Sisana", "54", "49", "left"),
        (125, "DDK", "54", "49", "left"),
        (139, "RNS_Sisana", "56", "51", "right"),
        (144, "BNM", "57", "52", "left"),
        (144, "BNT", "57", "52", "left"),
        (144, "RNK", "57", "52", "left"),
        (144, "RNS_Sisaikhara", "57", "52", "left"),
        (144, "RNS_Sisana", "57", "52", "left"),
        (144, "RKM", "57", "52", "left"),
        (144, "RKB", "57", "52", "left"),
        (144, "TkN", "57", "52", "left"),
        (144, "KkP", "57", "52", "left"),
        (144, "SkP", "57", "52", "left"),
        (144, "DKS", "57", "52", "left"),
        (144, "DDK", "57", "52", "left"),
        (144, "DGC", "57", "52", "left"),
        (144, "DkR", "57", "52", "left"),
        (144, "CCC", "57", "52", "left"),
        (145, "BNM", "57", "52", "right"),
        (145, "BNT", "57", "52", "right"),
        (145, "RNK", "57", "52", "right"),
        (145, "RNS_Sisaikhara", "57", "52", "right"),
        (145, "RNS_Sisana", "57", "52", "right"),
        (145, "RKM", "57", "52", "right"),
        (145, "RKB", "57", "52", "right"),
        (145, "TkN", "57", "52", "right"),
        (145, "KkP", "57", "52", "right"),
        (145, "SkP", "57", "52", "right"),
        (145, "DKS", "57", "52", "right"),
        (145, "DDK", "57", "52", "right"),
        (145, "DGC", "57", "52", "right"),
        (145, "DkR", "57", "52", "right"),
        (145, "CCC", "57", "52", "right"),
        (161, "CCC", "60", "55", "left"),
        (162, "CCC", "60", "55", "left"),
        (167, "CCC", "61", "56", "left"),
        (170, "CCC", "61", "56", "right"),
        (172, "RNS_Sisana", "61 / 62", "56 / 57", "right / left"),
        (178, "CCC", "63", "58", "left"),
        (179, "CCC", "63", "58", "left"),
        (184, "CCC", "64", "59", "left"),
        (186, "CCC", "64", "59", "right"),
        (188, "DkR", "64 / 65", "59 / 60", "right / left"),
        (193, "CCC", "65", "60", "right"),
        (196, "CCC", "66", "61", "left"),
        (199, "CCC", "66", "61", "right"),
        (206, "BNM", "68", "63", "left"),
        (206, "BNT", "68", "63", "left"),
        (206, "RNK", "68", "63", "left"),
        (206, "RNS_Sisaikhara", "68", "63", "left"),
        (206, "RNS_Sisana", "68", "63", "left"),
        (206, "RKM", "68", "63", "left"),
        (206, "RKB", "68", "63", "left"),
        (206, "TkN", "68", "63", "left"),
        (206, "KkP", "68", "63", "left"),
        (206, "SkP", "68", "63", "left"),
        (206, "DKS", "68", "63", "left"),
        (206, "DDK", "68", "63", "left"),
        (206, "DGC", "68", "63", "left"),
        (206, "DkR", "68", "63", "left"),
        (206, "CCC", "68", "63", "left"),
        (206, "HIN", "68", "63", "left"),
        (208, "CCC", "68", "63", "left"),
        (209, "CCC", "68", "63", "right"),
    ]
    assert rows[(1, "BNM")]["Manual_Transcription"] == "ʃʌɾiɾ / batʌn"
    assert rows[(1, "DGC")]["Manual_Transcription"] == "dẽh"
    assert rows[(2, "BNM")]["Manual_Transcription"] == "sɪɾ / muɖ"
    assert rows[(2, "RNS_Sisaikhara")]["Manual_Transcription"] == "mʊɖ"
    assert rows[(2, "CCC")]["Manual_Transcription"] == "muːɖ"
    assert rows[(2, "BNT")]["Manual_Transcription"] == "ɡʰopʌɖi"
    assert rows[(3, "DDK")]["Manual_Transcription"] == "bʰutla"
    assert rows[(4, "HIN")]["Manual_Transcription"] == "tʃehʌɾa / mũh"
    assert rows[(4, "RKB")]["Manual_Transcription"] == "muh"
    assert rows[(4, "RKB")]["Source_Qualifier"] == "used most"
    assert rows[(4, "RNS_Sisana")]["Manual_Transcription"] == "mʊh"
    assert rows[(5, "RKM")]["Manual_Transcription"] == "ãŋkʰi"
    assert rows[(5, "CCC")]["Manual_Transcription"] == "aiːkʰ"
    assert sum(int(row["Manual_Form_Count"]) for row in rows.values()) == 3850


def test_items_6_10_coordinates_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 6 <= item <= 10]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert {
        item: {(row["PDF_Page"], row["Printed_Page"], row["Column"])
               for (row_item, _), row in rows.items() if row_item == item}
        for item in range(6, 11)
    } == {
        6: {("32", "27", "left")},
        7: {("32", "27", "right")},
        8: {("32", "27", "right")},
        9: {("32", "27", "right")},
        10: {("33", "28", "left")},
    }
    assert rows[(8, "CCC")]["Manual_Transcription"] == "muːhʌ"
    assert rows[(8, "RNS_Sisaikhara")]["Manual_Transcription"] == "mũh"
    assert rows[(8, "RNS_Sisana")]["Manual_Transcription"] == "mʊh"
    assert rows[(9, "BNM")]["Manual_Transcription"] == "dand"
    assert rows[(9, "CCC")]["Manual_Transcription"] == "daːt"
    assert rows[(10, "CCC")]["Manual_Transcription"] == "dʒibʰi"
    qualified = [row for row in block if row["Source_Qualifier"]]
    assert len(qualified) == 12
    assert {row["Source_Qualifier"] for row in qualified} == {"(4)"}


def test_items_11_15_counts_page_breaks_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 11 <= item <= 15]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 97
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 89
    assert rows[(11, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "tʃutʃi / dudʰ / tʃutʃi"
    )
    assert rows[(11, "DDK")]["Manual_Transcription"] == "ɖuɖʰ"
    assert rows[(12, "BNM")]["Manual_Transcription"] == "peʈ"
    assert rows[(12, "CCC")]["Manual_Transcription"] == "peit"
    assert rows[(13, "TkN")]["Manual_Transcription"] == "hatʰ"
    assert rows[(13, "DKS")]["Manual_Transcription"] == "haʈʰ / pãtʃa"
    assert rows[(14, "DkR")]["Manual_Transcription"] == "ʈʰɪhũn"
    assert rows[(14, "DGC")]["Manual_Transcription"] == "ɡãɳʈʰ"
    assert rows[(15, "RNS_Sisaikhara")]["PDF_Page"] == "33 / 34"
    assert rows[(15, "RNS_Sisaikhara")]["Printed_Page"] == "28 / 29"
    assert rows[(15, "DDK")]["Manual_Transcription"] == "ɡʌɽɔɾi / ɡadi"
    assert rows[(15, "RKB")]["Source_Qualifier"] == "first response: (13)"
    assert rows[(15, "DkR")]["Source_Qualifier"] == "(13)"


def test_items_16_20_counts_page_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 16 <= item <= 20]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 87
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 81
    assert rows[(16, "RKB")]["Manual_Transcription"] == "ũŋɡʌɾi / ũŋɡʌli"
    assert rows[(16, "RNS_Sisaikhara")]["Manual_Transcription"] == "ʌ̃ŋɡʌɾja"
    assert rows[(16, "RNK")]["Manual_Transcription"] == "ũŋɡʌɾɪja"
    assert rows[(17, "DKS")]["Manual_Transcription"] == "nu / nuhũ"
    assert rows[(17, "DkR")]["Manual_Transcription"] == "nʊ̃"
    assert rows[(18, "RNS_Sisaikhara")]["Manual_Transcription"] == "ʈãŋɡ / pãv"
    assert rows[(18, "DGC")]["Manual_Transcription"] == "ɡoɾ / lat"
    assert rows[(19, "DkR")]["Manual_Transcription"] == "tʃokʌʈa / tʃʰala"
    assert rows[(19, "KkP")]["Manual_Transcription"] == "tʃʰutʌka"
    assert rows[(20, "DKS")]["PDF_Page"] == "34 / 35"
    assert rows[(20, "DKS")]["Printed_Page"] == "29 / 30"
    assert rows[(20, "DKS")]["Manual_Transcription"] == "hʌɖːi / ɖaŋɡʌɾ"
    assert rows[(20, "RKM")]["Manual_Transcription"] == "hʌɽːi"


def test_items_21_25_counts_blanks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 21 <= item <= 25]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 89
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 82
    assert rows[(21, "RNS_Sisaikhara")]["Manual_Transcription"] == "dɪl / kʌledʒa"
    assert rows[(21, "DGC")]["Manual_Transcription"] == "kʌledʒa / dʒiw"
    assert rows[(21, "DKS")]["Review_Status"] == "source_blank"
    assert rows[(22, "DKS")]["Manual_Transcription"] == "ɾʌɡʌt"
    assert rows[(23, "HIN")]["Manual_Transcription"] == "peʃab / mut"
    assert rows[(23, "CCC")]["Manual_Transcription"] == "mut / pʌsiena"
    assert rows[(24, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "hʌɡas / hʌɡʌdi / ɡuh"
    )
    assert rows[(24, "RNS_Sisana")]["Manual_Transcription"] == "hʌɡas / ɡuhu"
    assert rows[(24, "BNT")]["Manual_Transcription"] == "haɡʌna / hʌɡija"
    assert rows[(25, "RKM")]["Manual_Transcription"] == "ɡãõ"
    assert rows[(25, "CCC")]["Manual_Transcription"] == "ɡaːu"


def test_items_26_30_counts_page_breaks_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 26 <= item <= 30]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 98
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 91
    assert rows[(26, "RNK")]["PDF_Page"] == "35 / 36"
    assert rows[(26, "RNK")]["Printed_Page"] == "30 / 31"
    assert rows[(26, "BNM")]["Manual_Transcription"] == "ɡʰʌɾ / mʌkan"
    assert rows[(26, "BNM")]["Source_Qualifier"] == (
        "second response: (pukka house)"
    )
    assert rows[(26, "RKB")]["Manual_Transcription"] == "ɡʰʌɾ / mʌkʌn"
    assert rows[(27, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "tʃʌpːʌɾ / lintʌɾ"
    )
    assert rows[(27, "DGC")]["Manual_Transcription"] == "tʃʰʌt / tʃʰʌpʌɽa"
    assert rows[(27, "BNT")]["Manual_Transcription"] == "tʃʰʌpʌɾ / lʌɳɖʌɾ"
    assert rows[(28, "BNM")]["Column"] == "left / right"
    assert rows[(28, "TkN")]["Manual_Transcription"] == "kɪwaɽ"
    assert rows[(29, "RNK")]["Manual_Transcription"] == "kaɖʰɪja"
    assert rows[(29, "RNS_Sisana")]["Manual_Transcription"] == "kʌtʰɪja"
    assert rows[(30, "RNK")]["Manual_Transcription"] == "bʌɖnɪ"
    assert rows[(30, "RNS_Sisaikhara")]["Manual_Transcription"] == "bʌɽhʌni"
    assert rows[(30, "RKB")]["Source_Qualifier"] == (
        "first response: (small); second response: (big)"
    )
    assert rows[(30, "DKS")]["Manual_Transcription"] == "bʌɾʌni / sɪʈa"


def test_items_31_35_counts_page_breaks_blanks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 31 <= item <= 35]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 105
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 94
    assert rows[(31, "HIN")]["PDF_Page"] == "36 / 37"
    assert rows[(31, "HIN")]["Printed_Page"] == "31 / 32"
    assert rows[(31, "DDK")]["Manual_Transcription"] == "dokʌni / loɖʰa"
    assert rows[(31, "RNK")]["Manual_Transcription"] == "pʌtɪja / ɪmandʌsta"
    assert rows[(32, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "musʌɾa / kʊʈʌna"
    )
    assert rows[(32, "RKM")]["Manual_Transcription"] == "kʊʈʌna / kʊɽi"
    assert rows[(33, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(33, "RNK")]["Manual_Transcription"] == (
        "hʌtʰoɽi / hʌtʰoɽija"
    )
    assert rows[(33, "BNT")]["Manual_Transcription"] == "hʌtːɔɖa"
    assert rows[(34, "HIN")]["Manual_Form_Count"] == "5"
    assert rows[(34, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "tʃaku / tʃaku / hʌsija"
    )
    assert rows[(34, "DGC")]["Manual_Transcription"] == (
        "tʃʌkːu / tʃʌkːu / hʌsɪja / ɡʰuɾi"
    )
    assert rows[(35, "DGC")]["PDF_Page"] == "37 / 38"
    assert rows[(35, "DGC")]["Manual_Transcription"] == "bʌntʃeɾi / tegaɾi"
    assert rows[(35, "CCC")]["Manual_Transcription"] == "taŋi"
    assert rows[(35, "RKB")]["Manual_Transcription"] == "kʊdahari"


def test_items_36_40_counts_columns_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 36 <= item <= 40]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 90
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 83
    assert rows[(36, "BNM")]["Manual_Transcription"] == "ɾʌsːi / ɾʌsa"
    assert rows[(36, "DGC")]["Source_Qualifier"] == "second response: (thick)"
    assert rows[(37, "CCC")]["Manual_Transcription"] == "doaɾa / sut"
    assert rows[(38, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(38, "RNS_Sisana")]["Column"] == "right"
    assert rows[(39, "KkP")]["Manual_Transcription"] == "lʌʈa"
    assert rows[(39, "DGC")]["Manual_Transcription"] == "lʊɡʌɽa"
    assert rows[(40, "SkP")]["Manual_Transcription"] == "ãŋɡutʰi"
    assert rows[(40, "CCC")]["Manual_Transcription"] == "jʌŋɡuti"
    assert rows[(40, "KkP")]["Source_Qualifier"] == (
        "first response: (men's); second response: (women's)"
    )
    assert rows[(40, "RKM")]["Manual_Transcription"] == "mũdʌɾija"


def test_items_41_45_counts_page_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 41 <= item <= 45]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 90
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 82
    assert rows[(41, "BNT")]["PDF_Page"] == "38 / 39"
    assert rows[(41, "BNT")]["Printed_Page"] == "33 / 34"
    assert rows[(41, "BNT")]["Manual_Transcription"] == "suɾʌdʒ / din"
    assert rows[(41, "CCC")]["Manual_Transcription"] == "ɡʰam / beɾia"
    assert rows[(42, "RNS_Sisana")]["Manual_Transcription"] == "dʒõni"
    assert rows[(42, "DGC")]["Manual_Transcription"] == "ʌdʒeɾija"
    assert rows[(43, "RKB")]["Manual_Transcription"] == "badʌɾ / badʌl"
    assert rows[(43, "SkP")]["Manual_Transcription"] == "badɾi"
    assert rows[(44, "DkR")]["Manual_Transcription"] == "tõɾĩja"
    assert rows[(44, "SkP")]["Manual_Transcription"] == "tʌɾʌi + ja"
    assert rows[(45, "DkR")]["Source_Qualifier"] == "(46)"
    assert rows[(45, "TkN")]["Manual_Transcription"] == "bʌʃʌti"
    assert rows[(45, "KkP")]["Manual_Transcription"] == "bʌɾʃʌt / bʌɾsʌt"


def test_items_46_50_counts_blank_page_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 46 <= item <= 50]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 84
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 78
    assert rows[(46, "HIN")]["PDF_Page"] == "39 / 40"
    assert rows[(46, "HIN")]["Printed_Page"] == "34 / 35"
    assert rows[(46, "HIN")]["Manual_Transcription"] == "pani / dʒʌl"
    assert rows[(47, "RNS_Sisana")]["Source_Group_Labels"] == "1 1"
    assert rows[(47, "RNS_Sisana")]["Source_Qualifier"] == (
        "extra printed 1 after group label"
    )
    assert rows[(47, "BNM")]["Manual_Transcription"] == "nʌ̃ndi"
    assert rows[(47, "DDK")]["Manual_Transcription"] == "lʌɖijʌ"
    assert rows[(48, "RNS_Sisana")]["Manual_Transcription"] == "bʌdʌɾija"
    assert sum(bool(row["Source_Qualifier"]) for row in block if row["Item"] == "48") == 6
    assert rows[(49, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(49, "RNS_Sisana")]["Column"] == "right"
    assert rows[(49, "DGC")]["Manual_Transcription"] == "bʌdʌɾitʃʌmʌkʌtʰæ"
    assert rows[(50, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(50, "DGC")]["Manual_Transcription"] == (
        "indɾʌdʰʌnuʃ / dʰʌnuhi"
    )
    assert rows[(50, "DDK")]["Manual_Transcription"] == "ɖʰʌni"
    assert rows[(50, "DkR")]["Manual_Transcription"] == "ɾamʌtʃʌɾʌnketʃʰani"


def test_items_51_55_counts_blanks_page_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 51 <= item <= 55]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 83
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 76
    assert rows[(51, "BNT")]["Review_Status"] == "source_blank"
    assert rows[(51, "SkP")]["Manual_Transcription"] == "hʌw"
    assert rows[(51, "DGC")]["Manual_Transcription"] == "bʌjaɾ"
    assert rows[(52, "DDK")]["PDF_Page"] == "40 / 41"
    assert rows[(52, "DDK")]["Printed_Page"] == "35 / 36"
    assert rows[(52, "DDK")]["Manual_Transcription"] == "pʌtʰʌɾa / dũŋɡa"
    assert rows[(52, "BNM")]["Manual_Transcription"] == "pʌtʰːʌɾ"
    assert rows[(52, "DKS")]["Manual_Transcription"] == "pʌtʌjʌɾa"
    assert rows[(53, "RNS_Sisaikhara")]["Manual_Transcription"] == "ɾasta / ɾʌtːa"
    assert rows[(53, "RNK")]["Manual_Transcription"] == "ɾʌtːa / ɾʌtːa"
    assert rows[(53, "RNS_Sisana")]["Manual_Transcription"] == "ɾaha"
    assert rows[(53, "BNT")]["Review_Status"] == "source_blank"
    assert rows[(54, "HIN")]["Manual_Transcription"] == "balu / ɾet"
    assert rows[(55, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(55, "RNS_Sisana")]["Column"] == "right"
    assert rows[(55, "HIN")]["Manual_Transcription"] == "aɡ"
    assert rows[(55, "KkP")]["Manual_Transcription"] == "aɡi"


def test_items_56_60_counts_blanks_page_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 56 <= item <= 60]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 78
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 73
    assert rows[(56, "RNS_Sisaikhara")]["Manual_Transcription"] == "dʰũa"
    assert rows[(56, "RNS_Sisana")]["Manual_Transcription"] == "dʰuã"
    assert rows[(56, "CCC")]["Manual_Transcription"] == "dʰuːʌ"
    assert rows[(57, "RKB")]["Manual_Transcription"] == "bʰũa"
    assert rows[(57, "RKM")]["Manual_Transcription"] == "bʰʊa"
    assert rows[(57, "CCC")]["Manual_Transcription"] == "tʃʰaɖu"
    assert rows[(58, "RNS_Sisaikhara")]["PDF_Page"] == "41"
    assert rows[(58, "RNS_Sisana")]["PDF_Page"] == "42"
    assert rows[(58, "BNT")]["Manual_Transcription"] == "mʌʈːɪ"
    assert rows[(58, "KkP")]["Manual_Transcription"] == "kĩntʃʰa"
    assert rows[(58, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(59, "RNS_Sisaikhara")]["Manual_Transcription"] == "dʰudʰʌ̃ɾ"
    assert rows[(59, "RKM")]["Manual_Transcription"] == "dʰũdʰʌɾ"
    assert rows[(59, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(60, "RNS_Sisaikhara")]["Manual_Transcription"] == "sona"
    assert rows[(60, "RNS_Sisana")]["Manual_Transcription"] == "sono"
    assert rows[(60, "SkP")]["Manual_Transcription"] == "swan"


def test_items_61_65_counts_page_breaks_repeats_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 61 <= item <= 65]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 87
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 81
    assert rows[(61, "RNK")]["Column"] == "left / right"
    assert rows[(61, "RNK")]["Manual_Transcription"] == "peɖ / ɾukʰa"
    assert rows[(61, "SkP")]["Manual_Transcription"] == "ɾukʰːa"
    assert rows[(61, "CCC")]["Manual_Transcription"] == "ɡatʃʰ"
    assert rows[(62, "RNS_Sisaikhara")]["Manual_Transcription"] == "pʌʈːa"
    assert rows[(62, "RNS_Sisana")]["Manual_Transcription"] == "pʌtːa"
    assert rows[(62, "DGC")]["Manual_Transcription"] == "pʌʈija / pata"
    assert rows[(62, "DKS")]["Manual_Transcription"] == "pʌʈːija"
    assert rows[(63, "HIN")]["Manual_Transcription"] == "tʌna / dʒʌɾ"
    assert rows[(63, "DkR")]["Manual_Transcription"] == "hʌɡa / ɖahã"
    assert rows[(63, "RNS_Sisaikhara")]["Manual_Transcription"] == "hãŋɡa"
    assert rows[(63, "RNS_Sisana")]["Manual_Transcription"] == "hʌŋɡpa"
    assert rows[(64, "HIN")]["PDF_Page"] == "42"
    assert rows[(64, "DGC")]["PDF_Page"] == "43"
    assert rows[(64, "KkP")]["Manual_Transcription"] == "kaʈ / ɡaŋʈʰi"
    assert rows[(64, "CCC")]["Manual_Transcription"] == "kaːʈ"
    assert rows[(65, "HIN")]["Manual_Transcription"] == "pʰul"
    assert rows[(65, "RNS_Sisana")]["Manual_Transcription"] == "pʰula"


def test_items_66_70_counts_blanks_page_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 66 <= item <= 70]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=77, source_blank=3
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 79
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 73
    assert rows[(66, "DDK")]["Review_Status"] == "source_blank"
    assert rows[(66, "RKM")]["Manual_Transcription"] == "pʰʌɾa"
    assert rows[(67, "KkP")]["Manual_Transcription"] == "amb"
    assert rows[(68, "RNS_Sisaikhara")]["Manual_Transcription"] == "tʃʰijã"
    assert rows[(68, "RNS_Sisana")]["Manual_Transcription"] == "tʃʰija"
    assert rows[(68, "BNM")]["Manual_Transcription"] == "ɡeɾkibʰʌɾi"
    assert rows[(69, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(69, "SkP")]["Manual_Transcription"] == "ɡʊhõ"
    assert rows[(69, "TkN")]["Manual_Transcription"] == "ɡehu"
    assert rows[(70, "HIN")]["PDF_Page"] == "43 / 44"
    assert rows[(70, "HIN")]["Manual_Transcription"] == "dʒʌvaɾ / dʒɔ"
    assert rows[(70, "DkR")]["Manual_Transcription"] == "dʒolʌɾi"
    assert rows[(70, "RNS_Sisaikhara")]["Manual_Transcription"] == "dʒwaɾ / tʃʊɾi"
    assert rows[(70, "RNS_Sisana")]["Review_Status"] == "source_blank"
    assert rows[(70, "KkP")]["Manual_Transcription"] == "dʒoᵘ"
    assert rows[(70, "DKS")]["Manual_Transcription"] == "dʒaᵘ"


def test_items_71_75_counts_blanks_repeats_column_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 71 <= item <= 75]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 75
    assert rows[(71, "RNK")]["Manual_Transcription"] == "tʃamʌɾ / tʃawʌɾ"
    assert rows[(71, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(71, "SkP")]["Manual_Transcription"] == "tʃʌɔɾ"
    assert rows[(72, "CCC")]["Manual_Transcription"] == "alo"
    assert rows[(73, "HIN")]["Column"] == "left"
    assert rows[(73, "RNK")]["Column"] == "right"
    assert rows[(73, "CCC")]["Column"] == "left / right"
    assert rows[(73, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(73, "HIN")]["Manual_Transcription"] == "bæ̃ŋɡʌn"
    assert rows[(73, "RKB")]["Manual_Transcription"] == "bʌʈa"
    assert rows[(73, "DDK")]["Manual_Transcription"] == "bʰaɳʈa"
    assert rows[(74, "HIN")]["Manual_Transcription"] == "mũŋɡpʰʌli / mompʰʌli"
    assert rows[(74, "RNS_Sisaikhara")]["Manual_Transcription"] == "mumpʌɾi"
    assert rows[(74, "RNS_Sisana")]["Manual_Transcription"] == "mũpʰʌɾi"
    assert rows[(75, "DGC")]["Manual_Transcription"] == "miɾtʃi / miɾtʃa"
    assert rows[(75, "RKB")]["Manual_Transcription"] == "mitʃi"
    assert rows[(75, "CCC")]["Manual_Transcription"] == "maɾtʃa"


def test_items_76_80_counts_blanks_repeats_page_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 76 <= item <= 80]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 103
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 96
    assert rows[(76, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(76, "RKB")]["Manual_Transcription"] == "hʌɾʌdi"
    assert rows[(77, "DGC")]["Manual_Transcription"] == "lʌɾʌsʊn"
    assert rows[(77, "RKB")]["Manual_Transcription"] == "lʌhsun"
    assert rows[(78, "HIN")]["Manual_Form_Count"] == "3"
    assert rows[(78, "BNM")]["Manual_Transcription"] == (
        "pjadʒ / pjadʒ / pjadʒ / ɡʌɳʈʰi"
    )
    assert rows[(78, "RNS_Sisaikhara")]["Column"] == "left / right"
    assert rows[(78, "CCC")]["Manual_Transcription"] == "piadʒu"
    assert rows[(78, "KkP")]["Manual_Transcription"] == "pedʒ"
    assert rows[(79, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(79, "RKB")]["Manual_Transcription"] == "pʰulɡobi"
    assert rows[(80, "HIN")]["Manual_Transcription"] == "ʈʌmaʈʌɾ"
    assert rows[(80, "RKB")]["Manual_Transcription"] == "ʈimaʈʌɾ"
    assert rows[(80, "BNT")]["PDF_Page"] == "46"
    assert rows[(80, "BNT")]["Manual_Transcription"] == "ʈʌmʈʌmbʰʌʈa"
    assert rows[(80, "CCC")]["Manual_Transcription"] == "ɾambʰʌnʈa"


def test_items_81_85_counts_blanks_repeats_page_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 81 <= item <= 85]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=77, source_blank=3
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 96
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 90
    assert rows[(81, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(81, "DGC")]["Manual_Transcription"] == (
        "bʌndɡobʰi / patɡobʰi"
    )
    assert rows[(81, "DDK")]["Manual_Transcription"] == (
        "bʌndɡobʰi / ɡaɳʈʰɡobʰi"
    )
    assert rows[(82, "DGC")]["Manual_Transcription"] == "ʈel"
    assert rows[(82, "SkP")]["Manual_Transcription"] == "tjal"
    assert rows[(83, "RNK")]["Column"] == "left / right"
    assert rows[(83, "RNK")]["Manual_Transcription"] == "nun / nun"
    assert rows[(83, "BNM")]["Manual_Transcription"] == "non / non"
    assert rows[(83, "SkP")]["Manual_Transcription"] == "nwan"
    assert rows[(84, "TkN")]["Review_Status"] == "source_blank"
    assert rows[(84, "HIN")]["Manual_Transcription"] == "mãs / ɡoʃt"
    assert rows[(84, "DGC")]["Manual_Transcription"] == "ʃikaɾ"
    assert rows[(84, "RKB")]["Manual_Transcription"] == "sikaɾ / buʈi"
    assert rows[(84, "RKB")]["Source_Qualifier"] == (
        "second response: (small piece)"
    )
    assert rows[(85, "CCC")]["PDF_Page"] == "46 / 47"
    assert rows[(85, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(85, "RNS_Sisaikhara")]["Manual_Transcription"] == "tadʒõ"
    assert rows[(85, "RNS_Sisana")]["Manual_Transcription"] == "tadʒo"
    assert rows[(85, "DGC")]["Manual_Transcription"] == "moʈ"
    assert rows[(85, "DKS")]["Manual_Transcription"] == "muʈ"


def test_items_86_90_counts_blank_column_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 86 <= item <= 90]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 80
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 74
    assert rows[(86, "RNK")]["Manual_Transcription"] == "mʌtʃʰːi"
    assert rows[(86, "BNM")]["Manual_Transcription"] == "mʌtʃʰi"
    assert rows[(86, "DkR")]["Manual_Transcription"] == "mʌtʃʌhi"
    assert rows[(87, "BNM")]["Manual_Transcription"] == "mʊɡi"
    assert rows[(87, "RKM")]["Manual_Transcription"] == "mʊɾɡija"
    assert rows[(87, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(88, "RKM")]["Manual_Transcription"] == "ãɳɖa"
    assert rows[(88, "DDK")]["Manual_Transcription"] == "ãɳɽa"
    assert rows[(88, "DKS")]["Column"] == "right"
    assert rows[(89, "SkP")]["Manual_Transcription"] == "ɡʌjã"
    assert rows[(89, "RNS_Sisana")]["Manual_Transcription"] == "ɡʌjːã"
    assert rows[(90, "HIN")]["Manual_Transcription"] == "bʰæs / bʰæ̃s"
    assert rows[(90, "HIN")]["Source_Group_Labels"] == "1 / 1"
    assert rows[(90, "CCC")]["Manual_Transcription"] == "bʰæsi"
    assert rows[(90, "KkP")]["Manual_Transcription"] == "bʰaⁱsa"
    assert rows[(90, "DDK")]["Source_Group_Labels"] == "2"


def test_items_91_95_counts_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 91 <= item <= 95]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 85
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 79
    assert rows[(91, "KkP")]["PDF_Page"] == "47"
    assert rows[(91, "RKM")]["PDF_Page"] == "48"
    assert rows[(91, "RKM")]["Manual_Transcription"] == "dud"
    assert rows[(91, "CCC")]["Manual_Transcription"] == "dudʰa"
    assert rows[(92, "DDK")]["Manual_Transcription"] == "sĩŋ / kãʈa"
    assert rows[(92, "RKM")]["Manual_Transcription"] == "sĩŋɡ"
    assert rows[(92, "CCC")]["Manual_Transcription"] == "siŋ"
    assert rows[(93, "HIN")]["Manual_Transcription"] == "pũtʃʰ / ɖum"
    assert rows[(93, "DGC")]["Manual_Transcription"] == "putʃʰĩ"
    assert rows[(93, "RNS_Sisana")]["Manual_Transcription"] == "putʃʰija"
    assert rows[(94, "DGC")]["Column"] == "left"
    assert rows[(94, "DkR")]["Column"] == "right"
    assert rows[(94, "DGC")]["Manual_Transcription"] == "tʃʰʌɡɾija"
    assert rows[(94, "DDK")]["Manual_Transcription"] == "tʃʰʌɡʌɾi"
    assert rows[(95, "RKM")]["Manual_Transcription"] == "kuʈːa"
    assert rows[(95, "DGC")]["Manual_Transcription"] == "kʊkʌɾa / kʊkʌɾa"
    assert rows[(95, "CCC")]["Manual_Transcription"] == "kʊkʊɾu / kʊkʊɾu"


def test_items_96_100_counts_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 96 <= item <= 100]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 76
    assert rows[(96, "CCC")]["Manual_Transcription"] == "saːp"
    assert rows[(96, "DGC")]["Manual_Transcription"] == "sʌpuwa"
    assert rows[(96, "DkR")]["Manual_Transcription"] == "sapuã"
    assert rows[(97, "RKB")]["PDF_Page"] == "48"
    assert rows[(97, "TkN")]["PDF_Page"] == "49"
    assert rows[(97, "CCC")]["Manual_Transcription"] == "banʌɾ"
    assert rows[(98, "RKM")]["Manual_Transcription"] == "mʌtʃʰːʌɾ"
    assert rows[(98, "TkN")]["Manual_Transcription"] == "matʃʰʌɾ"
    assert rows[(99, "RNS_Sisaikhara")]["Manual_Transcription"] == "tʃĩtĩ"
    assert rows[(99, "RNS_Sisana")]["Manual_Transcription"] == "tʃiti"
    assert rows[(99, "KkP")]["Manual_Transcription"] == "tʃʰeⁱnti"
    assert rows[(100, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(100, "RNS_Sisana")]["Column"] == "right"
    assert rows[(100, "RKB")]["Manual_Transcription"] == "mʌkʌɾa / dʒara"
    assert rows[(100, "RKB")]["Source_Qualifier"] == "second response: (web)"
    assert rows[(100, "BNM")]["Manual_Transcription"] == "mʌkːʌɾi"
    assert rows[(100, "CCC")]["Manual_Transcription"] == "makara"
    assert rows[(100, "KkP")]["Manual_Transcription"] == "tʃʰiŋɡoɾa"


def test_items_101_105_counts_blanks_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 101 <= item <= 105]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 111
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 103
    assert rows[(101, "RNK")]["Manual_Transcription"] == "nãũ"
    assert rows[(101, "RKB")]["Manual_Transcription"] == "naõ"
    assert rows[(101, "TkN")]["Manual_Transcription"] == "não"
    assert rows[(101, "CCC")]["Manual_Transcription"] == "nʌːu"
    assert rows[(101, "KkP")]["Manual_Transcription"] == "naũ"
    assert rows[(102, "HIN")]["Manual_Transcription"] == "adʌmi / puruʃ"
    assert rows[(102, "BNM")]["Manual_Transcription"] == "adʌmi / amʌdi"
    assert rows[(102, "KkP")]["Manual_Transcription"] == "mʌnæ / log"
    assert rows[(103, "HIN")]["Manual_Transcription"] == "ɔɾʌt / stri"
    assert rows[(103, "HIN")]["PDF_Page"] == "49 / 50"
    assert rows[(103, "RNK")]["Manual_Transcription"] == "bʌtʃːʌɾ / bʌjːʌɾ"
    assert rows[(103, "KkP")]["Manual_Transcription"] == "meharu"
    assert rows[(103, "SkP")]["Manual_Transcription"] == "lʌdija"
    assert rows[(104, "RNS_Sisaikhara")]["Manual_Transcription"] == "balʌk / balʌk"
    assert rows[(104, "RNS_Sisana")]["Manual_Transcription"] == "balal"
    assert rows[(104, "BNT")]["Manual_Form_Count"] == "3"
    assert rows[(104, "DKS")]["Manual_Transcription"] == "lʌɖʌka / lʌɖʌka"
    assert rows[(104, "SkP")]["Manual_Transcription"] == "lʌɽʌka / lʌɽʌka"
    assert rows[(104, "DkR")]["Manual_Transcription"] == "loɽa"
    assert rows[(104, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(105, "HIN")]["Manual_Transcription"] == "pita / bap"
    assert rows[(105, "RNK")]["Manual_Transcription"] == "baba / baba / baba"
    assert rows[(105, "BNM")]["Manual_Transcription"] == "ʌbːa / bap"
    assert rows[(105, "DGC")]["Manual_Transcription"] == "bʌpːa / bʌpːa"
    assert rows[(105, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(105, "RNS_Sisana")]["Column"] == "right"
    assert rows[(105, "CCC")]["Review_Status"] == "source_blank"


def test_items_106_110_counts_blanks_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 106 <= item <= 110]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=76, source_blank=4
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 84
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 77
    assert rows[(106, "HIN")]["Manual_Transcription"] == "mata / mã"
    assert rows[(106, "BNM")]["Manual_Transcription"] == "abːu / ʌija"
    assert rows[(106, "RNS_Sisaikhara")]["Manual_Transcription"] == "ʌjːa"
    assert rows[(106, "RKB")]["Manual_Transcription"] == "ɔja"
    assert rows[(106, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(107, "HIN")]["Manual_Transcription"] == "bʌɾabʰai / dada"
    assert rows[(107, "RNK")]["Manual_Transcription"] == "dʌda"
    assert rows[(107, "DkR")]["Manual_Transcription"] == "dada / dadu"
    assert rows[(107, "DkR")]["PDF_Page"] == "50 / 51"
    assert rows[(107, "RKB")]["Manual_Transcription"] == "dʌta"
    assert rows[(107, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(108, "HIN")]["Manual_Transcription"] == "tʃʰoʈabʰai"
    assert rows[(108, "DkR")]["Manual_Transcription"] == "tʃʰuʈʌlibʰʌjːa"
    assert rows[(108, "SkP")]["Manual_Transcription"] == "bʰjːa"
    assert rows[(108, "CCC")]["Manual_Transcription"] == "ʌbaⁱja"
    assert rows[(108, "DDK")]["Manual_Transcription"] == "bʰaⁱwa"
    assert rows[(109, "BNT")]["Manual_Transcription"] == "didi / bʌhʌn"
    assert rows[(109, "DDK")]["Manual_Transcription"] == "ɖaɖi"
    assert rows[(109, "RKB")]["Manual_Transcription"] == "dɪdi"
    assert rows[(109, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(110, "RNS_Sisaikhara")]["Manual_Transcription"] == "bʌhʌn / lʌlo"
    assert rows[(110, "RNS_Sisana")]["Column"] == "right"
    assert rows[(110, "DGC")]["Manual_Transcription"] == "bʌhʌnija / babu"
    assert rows[(110, "RKM")]["Manual_Transcription"] == "bʌjinʌja"
    assert rows[(110, "RKB")]["Manual_Transcription"] == "lʌlːo"
    assert rows[(110, "BNT")]["Manual_Transcription"] == "ʌlːo"
    assert rows[(110, "CCC")]["Review_Status"] == "source_blank"


def test_items_111_115_counts_blanks_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 111 <= item <= 115]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=75, source_blank=5
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 86
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 79
    assert rows[(111, "HIN")]["Manual_Transcription"] == "beʈa / putra"
    assert rows[(111, "RNS_Sisaikhara")]["Manual_Transcription"] == "lɔɽa"
    assert rows[(111, "KkP")]["Manual_Transcription"] == "loɳɖa"
    assert rows[(111, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(112, "DGC")]["Manual_Transcription"] == "beʈi / lɔɽi"
    assert rows[(112, "RNS_Sisana")]["Manual_Transcription"] == "lɔɽija"
    assert rows[(112, "KkP")]["Manual_Transcription"] == "loɳɖia"
    assert rows[(112, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(113, "HIN")]["PDF_Page"] == "51"
    assert rows[(113, "BNT")]["PDF_Page"] == "52"
    assert rows[(113, "KkP")]["Manual_Transcription"] == "log / dulʌha / misaɾwa"
    assert rows[(113, "BNM")]["Manual_Transcription"] == "gʰʌɾwala"
    assert rows[(113, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(114, "RNK")]["Manual_Transcription"] == "bʌzdʒʌɾ / bʌjːʌɾ"
    assert rows[(114, "BNM")]["Manual_Transcription"] == "bʌjaɾ / gʰʌɾʌwali"
    assert rows[(114, "KkP")]["Manual_Transcription"] == "meharua"
    assert rows[(114, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(115, "BNT")]["Manual_Transcription"] == "balʌk / loɳɖe"
    assert rows[(115, "DDK")]["Manual_Transcription"] == "lɔɽa / tʃʰawa"
    assert rows[(115, "DKS")]["Manual_Transcription"] == "loɳɖa / tʃʰawa"
    assert rows[(115, "CCC")]["Review_Status"] == "source_blank"


def test_items_116_120_counts_blanks_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 116 <= item <= 120]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 85
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 79
    assert rows[(116, "BNM")]["Manual_Transcription"] == "lʌɽʌki / ʌlːo"
    assert rows[(116, "BNM")]["Column"] == "left / right"
    assert rows[(116, "RNS_Sisaikhara")]["Manual_Transcription"] == "lɔɽija"
    assert rows[(116, "KkP")]["Manual_Transcription"] == "loɳɖia"
    assert rows[(116, "SkP")]["Manual_Transcription"] == "tʃais"
    assert rows[(116, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(117, "DGC")]["Manual_Transcription"] == "ɖin"
    assert rows[(117, "CCC")]["Manual_Transcription"] == "din"
    assert rows[(118, "DGC")]["Manual_Transcription"] == "ɾaʈ"
    assert rows[(118, "CCC")]["Manual_Transcription"] == "ɾati"
    assert rows[(119, "HIN")]["Manual_Transcription"] == "sʊbʌh / sʌweɾa"
    assert rows[(119, "HIN")]["PDF_Page"] == "52 / 53"
    assert rows[(119, "BNM")]["Manual_Transcription"] == "tʌɖʌke"
    assert rows[(119, "DGC")]["Manual_Transcription"] == "sʌkaɾe / sʌkaɾe"
    assert rows[(119, "DkR")]["Manual_Transcription"] == "sʌkaɾ / vihan"
    assert rows[(120, "BNM")]["Manual_Transcription"] == "dupɔɾija"
    assert rows[(120, "TkN")]["Manual_Transcription"] == "dʊpʌhʌɾi"
    assert rows[(120, "RNS_Sisana")]["Manual_Transcription"] == "dʊpahʌɾi"
    assert rows[(120, "DkR")]["Manual_Transcription"] == "mintʃʰidʒun"
    assert rows[(120, "CCC")]["Review_Status"] == "source_blank"


def test_items_121_125_counts_blanks_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 121 <= item <= 125]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=77, source_blank=3
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 75
    assert rows[(121, "HIN")]["Manual_Transcription"] == "ʃam / saĩ"
    assert rows[(121, "HIN")]["Column"] == "left / right"
    assert rows[(121, "BNM")]["Manual_Transcription"] == "sãntʃ / etʃ"
    assert rows[(121, "RKB")]["Manual_Transcription"] == "sʌ̃dʒʰ"
    assert rows[(121, "DKS")]["Manual_Transcription"] == "sʌndʒa / sahidʒʊn"
    assert rows[(121, "CCC")]["Manual_Transcription"] == "saːdʒʰ"
    assert rows[(122, "HIN")]["Manual_Transcription"] == "kʌl"
    assert rows[(122, "DGC")]["Manual_Transcription"] == "kal"
    assert rows[(122, "CCC")]["Manual_Transcription"] == "kalu"
    assert rows[(123, "RNK")]["Manual_Transcription"] == "adʒ"
    assert rows[(123, "DKS")]["Manual_Transcription"] == "adʒʊ"
    assert rows[(124, "HIN")]["Source_Qualifier"] == "(122)"
    assert rows[(124, "DGC")]["Manual_Transcription"] == "kal"
    assert rows[(124, "DGC")]["PDF_Page"] == "53"
    assert rows[(124, "DkR")]["PDF_Page"] == "54"
    assert rows[(124, "CCC")]["Manual_Transcription"] == "andini"
    assert rows[(125, "RNS_Sisaikhara")]["Manual_Transcription"] == "hʌftah"
    assert rows[(125, "RNS_Sisana")]["Review_Status"] == "source_blank"
    assert rows[(125, "DkR")]["Manual_Transcription"] == "hʌpʈa"
    assert rows[(125, "DKS")]["Manual_Transcription"] == "hʌptʌh"
    assert rows[(125, "RKB")]["Manual_Transcription"] == "hʌptah / aʈʰʌdin"
    assert rows[(125, "RKB")]["Source_Qualifier"] == "second response: (used most)"
    assert rows[(125, "BNT")]["Review_Status"] == "source_blank"
    assert rows[(125, "DDK")]["Review_Status"] == "source_blank"


def test_items_126_130_counts_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 126 <= item <= 130]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 86
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 79
    assert rows[(126, "RNK")]["Manual_Transcription"] == "mʌhʌna"
    assert rows[(126, "TkN")]["Manual_Transcription"] == "mʌhɪna"
    assert rows[(126, "RNS_Sisana")]["Manual_Transcription"] == "mʌhina"
    assert rows[(126, "RKB")]["Manual_Transcription"] == "mahina"
    assert rows[(127, "HIN")]["Manual_Transcription"] == "sal / vʌɾʃ"
    assert rows[(127, "HIN")]["Column"] == "left / right"
    assert rows[(127, "DkR")]["Manual_Transcription"] == "bʌɾesdin"
    assert rows[(127, "DKS")]["Manual_Transcription"] == "sal / bʌɾʌs"
    assert rows[(128, "BNM")]["Manual_Transcription"] == "pʊɾana / bʌhutsalka"
    assert rows[(128, "RKB")]["Manual_Transcription"] == "purana"
    assert rows[(128, "DKS")]["Manual_Transcription"] == "puɾaɳa"
    assert rows[(129, "DkR")]["Manual_Transcription"] == "lʌbːa"
    assert rows[(129, "RKM")]["Manual_Transcription"] == "nʌ̃u"
    assert rows[(129, "KkP")]["Manual_Transcription"] == "nʌmːa"
    assert rows[(130, "HIN")]["Manual_Transcription"] == "ʌtʃʰːa / bʌɾija"
    assert rows[(130, "HIN")]["PDF_Page"] == "54 / 55"
    assert rows[(130, "RKB")]["Manual_Transcription"] == "atʃʰːa"
    assert rows[(130, "DkR")]["Manual_Transcription"] == "sʊgʰːʌɾ"
    assert rows[(130, "SkP")]["Manual_Transcription"] == "næŋg"
    assert rows[(130, "CCC")]["Manual_Transcription"] == "ɖol"


def test_items_131_135_counts_repeats_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 131 <= item <= 135]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 97
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 87
    assert rows[(131, "HIN")]["Manual_Transcription"] == "gʌnda / bekʌɾ / kʰʌɾab / bura"
    assert rows[(131, "SkP")]["Manual_Transcription"] == "tʃʰɪʈɔn"
    assert rows[(131, "DkR")]["Manual_Transcription"] == "gʌndhʌjʌna"
    assert rows[(131, "KkP")]["Manual_Transcription"] == "bʰuhʌɾ / mælʌha"
    assert rows[(131, "KkP")]["Source_Qualifier"] == (
        "first response: (person); second response: (object)"
    )
    assert rows[(132, "HIN")]["Manual_Transcription"] == "bʰiga / gila"
    assert rows[(132, "BNM")]["Manual_Transcription"] == "bʰiga / bʰidʒ"
    assert rows[(132, "DkR")]["Manual_Transcription"] == "bʰidʒʌgil"
    assert rows[(133, "HIN")]["Column"] == "left"
    assert rows[(133, "DGC")]["Column"] == "right"
    assert rows[(133, "RKM")]["Manual_Transcription"] == "sukhʌna"
    assert rows[(133, "DkR")]["Manual_Transcription"] == "sogʌgil"
    assert rows[(134, "RNK")]["Manual_Transcription"] == "lʌmbõ"
    assert rows[(134, "DDK")]["Manual_Transcription"] == "lamma / dʰẽɖ"
    assert rows[(134, "CCC")]["Manual_Transcription"] == "nʌmʌhaɾa"
    assert rows[(135, "RNK")]["Manual_Transcription"] == "tʃʰoʈo / tʃʰoʈo"
    assert rows[(135, "RNK")]["PDF_Page"] == "55 / 56"
    assert rows[(135, "KkP")]["Manual_Transcription"] == "tʃʰoʈ / tʃoʈimoti"
    assert rows[(135, "DGC")]["Manual_Transcription"] == "tʃʰoʈʌmoʈ"
    assert rows[(135, "SkP")]["Manual_Transcription"] == "tʃʌwaʈ"


def test_items_136_140_counts_blank_breaks_qualifier_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 136 <= item <= 140]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 88
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 83
    assert rows[(136, "BNM")]["Manual_Transcription"] == "gaɾʌm / tʌʈːi"
    assert rows[(136, "BNM")]["Source_Qualifier"] == "first response: (weather)"
    assert rows[(136, "RKM")]["Manual_Transcription"] == "tʌʈːo"
    assert rows[(137, "HIN")]["Manual_Transcription"] == "ʈʰʌɳɖa"
    assert rows[(137, "CCC")]["Manual_Transcription"] == "tʌɳɖʰa / dʒaɖ"
    assert rows[(137, "RKB")]["Manual_Transcription"] == "dʒudo"
    assert rows[(138, "BNM")]["Column"] == "left / right"
    assert rows[(138, "RNS_Sisana")]["Manual_Transcription"] == "dʌhino"
    assert rows[(139, "RNS_Sisaikhara")]["Manual_Transcription"] == "bão / ɖibʌno"
    assert rows[(139, "RNS_Sisana")]["Review_Status"] == "source_blank"
    assert rows[(139, "CCC")]["Manual_Transcription"] == "bajaː / lʌdʌɖi"
    assert rows[(140, "RNS_Sisaikhara")]["Manual_Transcription"] == "dʒʰɔno"
    assert rows[(140, "SkP")]["Manual_Transcription"] == "ʈʰɔɽe"
    assert rows[(140, "RKB")]["Manual_Transcription"] == "ɖʰiŋgai"


def test_items_141_145_counts_blanks_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 141 <= item <= 145]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=50, source_blank=30
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 50
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 45
    assert rows[(141, "HIN")]["PDF_Page"] == "56"
    assert rows[(141, "RNS_Sisaikhara")]["PDF_Page"] == "57"
    assert rows[(141, "TkN")]["Manual_Transcription"] == "duɾ"
    assert rows[(141, "DKS")]["Manual_Transcription"] == "dʊɾ"
    assert rows[(141, "CCC")]["Manual_Transcription"] == "tʌnau"
    assert rows[(142, "DGC")]["Manual_Transcription"] == "bʰaɽi"
    assert rows[(142, "DkR")]["Manual_Transcription"] == "bʰaɾi"
    assert rows[(142, "RKB")]["Manual_Transcription"] == "bʌdo"
    assert rows[(142, "CCC")]["Manual_Transcription"] == "dʒabʌɖe"
    assert rows[(143, "DDK")]["Manual_Transcription"] == "tʃʰuʈinʌg"
    assert rows[(143, "DkR")]["Manual_Transcription"] == "tʃʰoʈimoʈi"
    assert rows[(143, "SkP")]["Manual_Transcription"] == "tʃʰʌwaʈ"
    assert sum(row["Source_Qualifier"] == "(135)" for row in block) == 9
    assert rows[(144, "HIN")]["Manual_Transcription"] == "bʰaɾi"
    assert rows[(145, "HIN")]["Manual_Transcription"] == "hʌlka"
    assert all(
        rows[(item, site)]["Review_Status"] == "source_blank"
        for item in (144, 145) for site in MOD.TARGETS
    )


def test_items_146_150_counts_repeats_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 146 <= item <= 150]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 76
    assert rows[(146, "DGC")]["Manual_Transcription"] == "ʊpːʌɾ"
    assert rows[(146, "RKB")]["Manual_Transcription"] == "upːʌɾ"
    assert rows[(146, "CCC")]["Manual_Transcription"] == "upːiɾi"
    assert rows[(147, "DKS")]["Manual_Transcription"] == "ʈʌɾe"
    assert rows[(147, "DkR")]["Manual_Transcription"] == "tæɾe"
    assert rows[(147, "CCC")]["Manual_Transcription"] == "eʈːo"
    assert rows[(148, "BNM")]["Manual_Transcription"] == "seta / bʰuɾo"
    assert rows[(148, "BNM")]["PDF_Page"] == "57 / 58"
    assert rows[(148, "DGC")]["Manual_Transcription"] == "ʊdʒːʌɾ"
    assert rows[(148, "DkR")]["Manual_Transcription"] == "ʊɖːal"
    assert rows[(148, "DDK")]["Manual_Transcription"] == "uɖːaɾ"
    assert rows[(148, "CCC")]["Manual_Transcription"] == "goɾʌhʌɾ"
    assert rows[(149, "DGC")]["Manual_Transcription"] == "kʌɾija"
    assert rows[(149, "CCC")]["Manual_Transcription"] == "kʌɾiʌ"
    assert {rows[(150, site)]["Manual_Transcription"] for site in MOD.load_registry()} == {"lal"}


def test_items_151_155_counts_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 151 <= item <= 155]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 80
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 75
    assert {rows[(151, site)]["Manual_Transcription"] for site in MOD.load_registry()} == {"ek"}
    assert rows[(151, "RNS_Sisana")]["Column"] == "left"
    assert rows[(151, "CCC")]["Column"] == "right"
    assert rows[(152, "RNK")]["Manual_Transcription"] == "dui"
    assert rows[(152, "TkN")]["Manual_Transcription"] == "dʊi"
    assert rows[(152, "RNS_Sisana")]["Manual_Transcription"] == "dʊi"
    assert rows[(153, "RNK")]["Manual_Transcription"] == "tin"
    assert rows[(153, "DGC")]["Manual_Transcription"] == "ʈin"
    assert rows[(154, "RNK")]["PDF_Page"] == "58"
    assert rows[(154, "CCC")]["PDF_Page"] == "59"
    assert {rows[(154, site)]["Manual_Transcription"] for site in MOD.load_registry()} == {"tʃaɾ"}
    assert rows[(155, "RNS_Sisaikhara")]["Manual_Transcription"] == "pãntʃ"
    assert rows[(155, "RNK")]["Manual_Transcription"] == "patʃ"
    assert rows[(155, "CCC")]["Manual_Transcription"] == "paːtʃ"


def test_items_156_160_counts_breaks_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 156 <= item <= 160]
    assert len(block) == 80
    assert {row["Review_Status"] for row in block} == {"attested"}
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 80
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 75
    assert rows[(156, "RKM")]["Manual_Transcription"] == "tʃʰʌⁱ"
    assert rows[(156, "DDK")]["Manual_Transcription"] == "tʃʰɔᶸ"
    assert rows[(156, "CCC")]["Manual_Transcription"] == "tʃʰo"
    assert rows[(157, "DGC")]["Manual_Transcription"] == "saʈ"
    assert rows[(157, "RNK")]["Column"] == "left"
    assert rows[(157, "DDK")]["Column"] == "right"
    assert rows[(158, "RNS_Sisaikhara")]["Manual_Transcription"] == "aʈʰ"
    assert rows[(158, "CCC")]["Manual_Transcription"] == "at"
    assert rows[(159, "RNK")]["Manual_Transcription"] == "nɔ"
    assert rows[(159, "CCC")]["Manual_Transcription"] == "nou"
    assert rows[(160, "RNK")]["PDF_Page"] == "59"
    assert rows[(160, "DGC")]["PDF_Page"] == "60"
    assert rows[(160, "DGC")]["Manual_Transcription"] == "ɖʌs"
    assert rows[(160, "CCC")]["Manual_Transcription"] == "das"


def test_items_161_165_counts_blanks_break_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 161 <= item <= 165]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 79
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 74
    assert rows[(161, "BNM")]["Manual_Transcription"] == "gjaɾʌh"
    assert rows[(161, "SkP")]["Manual_Transcription"] == "ɪgjaɾʌh"
    assert rows[(161, "RKB")]["Manual_Transcription"] == "gaɾʌh"
    assert rows[(161, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(162, "SkP")]["Manual_Transcription"] == "baɾʌhʌɾ"
    assert rows[(162, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(163, "CCC")]["Column"] == "left"
    assert rows[(163, "KkP")]["Column"] == "right"
    assert {rows[(163, site)]["Manual_Transcription"] for site in MOD.load_registry()} == {"bis"}
    assert rows[(164, "RNS_Sisaikhara")]["Manual_Transcription"] == "sɔ"
    assert rows[(164, "CCC")]["Manual_Transcription"] == "sai"
    assert rows[(165, "RNK")]["Manual_Transcription"] == "kɔːn"
    assert rows[(165, "RNS_Sisana")]["Manual_Transcription"] == "kɔːn"
    assert rows[(165, "DGC")]["Manual_Transcription"] == "kʌʊn / ke"
    assert rows[(165, "DGC")]["Source_Group_Labels"] == "1 / 3"
    assert rows[(165, "DKS")]["Manual_Transcription"] == "ke"
    assert rows[(165, "DKS")]["Source_Group_Labels"] == "3"


def test_items_166_170_counts_blanks_break_qualifier_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 166 <= item <= 170]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 90
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 84
    assert rows[(166, "RKB")]["PDF_Page"] == "60"
    assert rows[(166, "KkP")]["PDF_Page"] == "61"
    assert rows[(166, "CCC")]["Manual_Transcription"] == "kʌtʰi"
    assert rows[(167, "RNS_Sisaikhara")]["Manual_Transcription"] == "kʌhã"
    assert rows[(167, "RNS_Sisana")]["Manual_Transcription"] == "kʌhãko"
    assert rows[(167, "RKM")]["Manual_Transcription"] == "kʌha"
    assert rows[(167, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(168, "DGC")]["Manual_Transcription"] == "kʌb / kʌhija"
    assert rows[(168, "DGC")]["Source_Qualifier"] == "second response: (future)"
    assert rows[(168, "CCC")]["Manual_Transcription"] == "dʒʌb"
    assert rows[(169, "DGC")]["Manual_Transcription"] == "kʌtʌna / kæitʰo"
    assert rows[(169, "DGC")]["Column"] == "left / right"
    assert rows[(169, "SkP")]["Manual_Transcription"] == "kʌtːa / kʌtːa"
    assert rows[(169, "RNS_Sisana")]["Column"] == "right"
    assert rows[(170, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "kɔnsitʌɾʌhʌko / konso / konsitʌɾʌhʌko"
    )
    assert rows[(170, "RNS_Sisana")]["Manual_Transcription"] == "kɔnsi"
    assert rows[(170, "KkP")]["Manual_Transcription"] == "kæse / kæsʌn"
    assert rows[(170, "DkR")]["Manual_Transcription"] == "kʌtʌɾaɾʌ̃ŋke"
    assert rows[(170, "CCC")]["Review_Status"] == "source_blank"


def test_items_171_175_counts_blank_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 171 <= item <= 175]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 88
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 80
    assert rows[(171, "RNS_Sisaikhara")]["Manual_Transcription"] == "dʒʌw"
    assert rows[(171, "RNS_Sisana")]["Manual_Transcription"] == "dʒʌ"
    assert rows[(171, "KkP")]["Manual_Transcription"] == "i"
    assert rows[(172, "BNM")]["PDF_Page"] == "61"
    assert rows[(172, "RNK")]["PDF_Page"] == "62"
    assert rows[(172, "RNS_Sisaikhara")]["Manual_Transcription"] == "bo / hʊn"
    assert rows[(172, "RNS_Sisana")]["Review_Status"] == "source_blank"
    assert "duplicate source code RNS" in rows[(172, "RNS_Sisana")]["Uncertainty"]
    assert rows[(172, "DGC")]["Manual_Transcription"] == "vohe / u"
    assert rows[(172, "TkN")]["Source_Qualifier"] == "(171)"
    assert rows[(173, "BNM")]["Manual_Transcription"] == "jẽ / ɪtna"
    assert rows[(173, "DGC")]["Manual_Transcription"] == "i / ajne"
    assert rows[(173, "DGC")]["Source_Qualifier"] == "first response: (171)"
    assert rows[(173, "DDK")]["Manual_Transcription"] == "i / tæ"
    assert rows[(174, "DDK")]["Manual_Transcription"] == "ʊ"
    assert rows[(174, "DkR")]["Source_Qualifier"] == "(171)"
    assert rows[(175, "HIN")]["Manual_Transcription"] == "eksʌman / ekse / sʌman"
    assert rows[(175, "HIN")]["Source_Group_Labels"] == "1 / 3 / b"
    assert rows[(175, "DGC")]["Manual_Transcription"] == "ekːægʰʌs / ekːæmeɾ"
    assert rows[(175, "KkP")]["Manual_Transcription"] == "eketaɾ"
    assert rows[(175, "CCC")]["Manual_Transcription"] == "ɾitto"
    assert rows[(175, "CCC")]["Source_Qualifier"] == "(alike)"


def test_items_176_180_counts_blanks_break_source_code_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 176 <= item <= 180]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 98
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 85
    assert rows[(176, "HIN")]["Manual_Transcription"] == "ʌlʌgʌlʌg / fʌɾʌk / bʰinː"
    assert rows[(176, "DDK")]["Manual_Transcription"] == "ʌligẽʌligẽ"
    assert rows[(176, "DGC")]["Manual_Transcription"] == "ʌlʌgeʌlʌge / dusʌɾdusʌɾ"
    assert rows[(176, "DKS")]["Source_Code"] == "DK"
    assert rows[(176, "KkP")]["Manual_Transcription"] == (
        "ɔɾeʈaɾɔɾeʈaɾ / ɔɾeʈʰaɾɔɾeʈʰaɾ"
    )
    assert rows[(177, "DKS")]["PDF_Page"] == "62 / 63"
    assert rows[(177, "RNS_Sisaikhara")]["Manual_Transcription"] == "puɾo"
    assert rows[(177, "RNS_Sisana")]["Manual_Transcription"] == "sʌb"
    assert rows[(177, "CCC")]["Manual_Transcription"] == "ond̪i"
    assert rows[(178, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(178, "DDK")]["Manual_Transcription"] == "pʰuʈʌlw̃"
    assert rows[(178, "DKS")]["Manual_Transcription"] == "tutgaijja"
    assert rows[(178, "RKB")]["Manual_Transcription"] == "dʰuʈo"
    assert rows[(179, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(179, "HIN")]["Manual_Transcription"] == (
        "kʊtʃʰ / tʰoɾi / t̪ʰoɾa / kʌm"
    )
    assert rows[(179, "DGC")]["Manual_Transcription"] == "tʰoɾiek / tʰɔɾewun"
    assert rows[(179, "SkP")]["Manual_Transcription"] == "tʌndjaka"
    assert rows[(179, "RNS_Sisana")]["Manual_Transcription"] == "dʒʌɾʌjegʰaj"
    assert rows[(180, "BNM")]["Manual_Transcription"] == "bʌhʊt"
    assert rows[(180, "CCC")]["Manual_Transcription"] == "bʌhut"
    assert rows[(180, "DGC")]["Manual_Transcription"] == "barider"
    assert rows[(180, "RKM")]["Manual_Transcription"] == "bʌɖadʒoi"
    assert rows[(180, "RNS_Sisana")]["Manual_Transcription"] == "bʌɾadʒo"


def test_items_181_185_counts_blank_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 181 <= item <= 185]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 135
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 125
    assert rows[(181, "DkR")]["Manual_Transcription"] == "sʌɾʌdʒ"
    assert rows[(181, "CCC")]["Manual_Transcription"] == "dʒʌmai / bʰare"
    assert rows[(181, "RKB")]["Source_Qualifier"] == "(176)"
    assert rows[(182, "RNS_Sisaikhara")]["PDF_Page"] == "63"
    assert rows[(182, "RNS_Sisana")]["PDF_Page"] == "64"
    assert rows[(182, "DDK")]["Manual_Transcription"] == "kʰadʒʰʌtʌe / kʰa"
    assert rows[(182, "DDK")]["Source_Qualifier"] == (
        "second response followed by literal ellipsis (...)"
    )
    assert rows[(183, "CCC")]["Manual_Transcription"] == "kʌtʌi / tokʌi"
    assert rows[(183, "RKM")]["Manual_Transcription"] == "kaʈo / kaʈoɾʌhe"
    assert rows[(184, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(184, "RKM")]["Manual_Transcription"] == "bʰukʰo / bʰukʰorʌ"
    assert rows[(184, "RNS_Sisana")]["Manual_Transcription"] == (
        "bʰukʰo / bʰukʰorʌhʊ"
    )
    assert rows[(185, "RNS_Sisaikhara")]["Column"] == "left"
    assert rows[(185, "RNS_Sisana")]["Column"] == "right"
    assert rows[(185, "RNS_Sisana")]["Manual_Transcription"] == "pile / pilʊhʊ̃"


def test_items_186_190_counts_blanks_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 186 <= item <= 190]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 140
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 130
    assert rows[(186, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(186, "TkN")]["Manual_Transcription"] == "pjasohʊ̃ / pjaso"
    assert rows[(186, "DKS")]["Manual_Transcription"] == "pjas / pjasʌn"
    assert "literal ellipsis" in rows[(186, "DKS")]["Source_Qualifier"]
    assert rows[(186, "RKM")]["Manual_Transcription"] == "pjaso / pjasoɾʌhʌgʊ"
    assert rows[(187, "RKB")]["Manual_Transcription"] == "sojdʒa / sojjo"
    assert rows[(187, "RNS_Sisana")]["Manual_Transcription"] == "sojdʒa / sotɾʌhẽ"
    assert rows[(187, "DKS")]["Manual_Transcription"] == "sutdʒa / sutʌgʌjjilʌs"
    assert rows[(188, "DkR")]["Review_Status"] == "source_blank"
    assert rows[(188, "DkR")]["PDF_Page"] == "64 / 65"
    assert rows[(188, "RKM")]["Manual_Transcription"] == "ledʒdʒaːleʈõɾʌhõ"
    assert rows[(188, "RKM")]["Manual_Form_Count"] == "1"
    assert rows[(188, "SkP")]["Manual_Transcription"] == "ʊɽʌɾdʒa / ʊɽʌɾʌl"
    assert rows[(188, "CCC")]["Manual_Transcription"] == "ulʈʌi / pulʈʌi"
    assert rows[(188, "CCC")]["Source_Group_Labels"] == "2 / 4"
    assert rows[(188, "DKS")]["Source_Qualifier"] == "second response: (187)"
    assert rows[(189, "CCC")]["Manual_Transcription"] == "betʌi / besʌi"
    assert rows[(189, "DDK")]["Manual_Transcription"] == "bæʈtaji / bæʈʌlɾʌhʊ"
    assert rows[(189, "DKS")]["Manual_Transcription"] == "bæʈʰai / bæʈʰʌgʌjjinʊ"
    assert rows[(190, "RNK")]["Manual_Transcription"] == "dæjʌdæ / dʌiɾʌhæ"
    assert rows[(190, "RNS_Sisana")]["Manual_Transcription"] == "dejʌde / dejʌdɪ"
    assert rows[(190, "RKM")]["Manual_Transcription"] == "dæidæ / dæidæi"


def test_items_191_195_counts_blank_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 191 <= item <= 195]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=79, source_blank=1
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 145
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 135
    assert rows[(191, "DGC")]["Manual_Transcription"] == "dʒʌla / bʌɾʌta / bʌɾʌtʰæ̃"
    assert rows[(191, "DGC")]["Column"] == "left / right"
    assert rows[(191, "DKS")]["Manual_Transcription"] == "bʌɾʌta / dʒʌɾʌgʌjji"
    assert rows[(191, "TkN")]["Manual_Transcription"] == "pʌdʒʌɾʌtʰæ̃ / pʌ"
    assert "literal ellipsis" in rows[(191, "TkN")]["Source_Qualifier"]
    assert rows[(192, "TkN")]["Manual_Transcription"] == "mʌrɾʌhohæ̃ / mʌr"
    assert rows[(192, "RKB")]["Manual_Transcription"] == "mʌrʌnlego / mʌrɪgao"
    assert rows[(193, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(193, "DkR")]["Source_Qualifier"] == "second response: (192)"
    assert rows[(193, "KkP")]["Manual_Transcription"] == "mardarʌl"
    assert rows[(194, "RNS_Sisaikhara")]["Manual_Transcription"] == "ʊɽʌɾʌu / ʊɽgʌji"
    assert rows[(194, "RNS_Sisaikhara")]["PDF_Page"] == "65"
    assert rows[(194, "RNS_Sisana")]["Manual_Transcription"] == "uɾːɪhæ̃ / uɾːɪrʌhẽ"
    assert rows[(194, "RNS_Sisana")]["PDF_Page"] == "66"
    assert rows[(194, "TkN")]["Manual_Transcription"] == "ʊɽɾʌhihæ̃ʊɽʌt"
    assert rows[(194, "TkN")]["Manual_Form_Count"] == "1"
    assert rows[(194, "DkR")]["Manual_Transcription"] == "ʊɽʌta / ʊɾʌt"
    assert rows[(195, "RKB")]["Manual_Form_Count"] == "4"
    assert rows[(195, "DDK")]["Manual_Form_Count"] == "4"
    assert rows[(195, "DGC")]["Manual_Transcription"] == "negʌŋʌt / ɾæ̃ŋʌg"
    assert rows[(195, "DkR")]["Manual_Transcription"] == "næɽ / næɽʌl"
    assert rows[(195, "CCC")]["Manual_Transcription"] == "bulʌi / gʰɪmʌi"
    assert rows[(195, "RKM")]["Source_Qualifier"] == (
        "second response followed by literal ellipsis (...)"
    )


def test_items_196_200_counts_blanks_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 196 <= item <= 200]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=78, source_blank=2
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 144
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 134
    assert rows[(196, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(196, "RKB")]["Manual_Transcription"] == (
        "dɔdo / dɔda / bʰadʒ / bʰadʒʌt"
    )
    assert rows[(196, "RNS_Sisaikhara")]["Manual_Transcription"] == (
        "dɔrre / bʰadʒ / bʰadʒo / bʰadʒgʌʊ / dʒʰʌno"
    )
    assert rows[(196, "TkN")]["Manual_Transcription"] == "dɔrdʒa / dɔrrʌho"
    assert "literal ellipsis" in rows[(196, "RKM")]["Source_Qualifier"]
    assert rows[(197, "DKS")]["Manual_Transcription"] == "dʒao / gʌjigɪl"
    assert "literal colon" in rows[(197, "DKS")]["Source_Qualifier"]
    assert rows[(197, "KkP")]["Source_Qualifier"] == "response followed by (past)"
    assert rows[(197, "RNS_Sisaikhara")]["Source_Group_Labels"] == "2 / 2 / 3"
    assert rows[(198, "BNM")]["Manual_Transcription"] == "ʊlʌ̃gao"
    assert "parentheses" in rows[(198, "BNM")]["Source_Qualifier"]
    assert rows[(198, "RNS_Sisana")]["Manual_Transcription"] == "ajːdʒa / ajʌgʊ"
    assert rows[(199, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(199, "RKM")]["Manual_Transcription"] == "kʌh / kʌhi"
    assert rows[(200, "HIN")]["PDF_Page"] == "66"
    assert rows[(200, "RNK")]["PDF_Page"] == "67"
    assert rows[(200, "DkR")]["Manual_Transcription"] == "sun / sunːʊ"
    assert rows[(200, "RNS_Sisana")]["Manual_Transcription"] == "sʊnle / sʊnʌtrʌhẽ"


def test_items_201_205_counts_break_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 201 <= item <= 205]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(attested=80)
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 95
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 88
    assert rows[(201, "DkR")]["Manual_Transcription"] == "dekʰun / her"
    assert "(look)" in rows[(201, "DkR")]["Source_Qualifier"]
    assert "(see)" in rows[(201, "DkR")]["Source_Qualifier"]
    assert rows[(201, "DGC")]["Manual_Transcription"] == "dekʰnu / her"
    assert rows[(201, "CCC")]["Manual_Transcription"] == "herʌi"
    assert rows[(201, "RKB")]["Manual_Transcription"] == "dekhʌto / dekhʌlɔ"
    assert rows[(202, "CCC")]["Manual_Transcription"] == "muːi"
    assert rows[(202, "DDK")]["Manual_Transcription"] == "mʌi"
    assert rows[(203, "HIN")]["Manual_Transcription"] == "tʊm / tu"
    assert rows[(203, "HIN")]["Source_Group_Labels"] == "1 / 2"
    assert rows[(203, "DGC")]["Manual_Transcription"] == "tʌĩ"
    assert rows[(203, "DDK")]["Manual_Transcription"] == "tæ̃i"
    assert rows[(204, "DKS")]["Manual_Transcription"] == "ʈʊ"
    assert rows[(204, "DDK")]["Manual_Transcription"] == "tũ"
    assert rows[(204, "CCC")]["Manual_Transcription"] == "jʌpʌnahike"
    assert rows[(205, "RNS_Sisana")]["Source_Qualifier"] == "response followed by (174)"
    assert rows[(205, "CCC")]["PDF_Page"] == "68"
    assert rows[(205, "CCC")]["Printed_Page"] == "63"
    assert rows[(205, "CCC")]["Manual_Transcription"] == "ua"


def test_items_206_210_counts_blanks_qualifiers_and_visual_regressions():
    rows = {(int(row["Item"]), row["Site_Key"]): row for row in MOD.load_cells()}
    block = [row for (item, _), row in rows.items() if 206 <= item <= 210]
    assert len(block) == 80
    assert Counter(row["Review_Status"] for row in block) == Counter(
        attested=62, source_blank=18
    )
    assert sum(int(row["Manual_Form_Count"]) for row in block) == 69
    assert sum(
        int(row["Manual_Form_Count"]) for row in block if row["Scope"] == "target"
    ) == 65
    assert all(rows[(206, site)]["Review_Status"] == "source_blank" for site in MOD.load_registry())
    assert rows[(207, "RNS_Sisaikhara")]["Manual_Transcription"] == "hʌm / hʌmsʌb"
    assert rows[(207, "RNS_Sisaikhara")]["Source_Group_Labels"] == "1 / 2"
    assert rows[(207, "DDK")]["Manual_Transcription"] == "hʌmʌrẽ / hʌmʌrẽsʌb"
    assert rows[(207, "CCC")]["Manual_Transcription"] == "hamara"
    assert rows[(208, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(208, "DkR")]["Source_Qualifier"] == "response followed by (202)"
    assert rows[(209, "CCC")]["Review_Status"] == "source_blank"
    assert rows[(209, "RNS_Sisaikhara")]["Manual_Transcription"] == "tʊm / tumlog"
    assert rows[(209, "KkP")]["Manual_Transcription"] == "tum / tumʌreh"
    assert rows[(209, "DGC")]["Manual_Transcription"] == "tohʌre / tɔhi"
    assert rows[(209, "DKS")]["Manual_Transcription"] == "ʈureh"
    assert rows[(210, "RNS_Sisaikhara")]["Manual_Transcription"] == "ve / vou"
    assert rows[(210, "RKM")]["Manual_Transcription"] == "voʊ"
    assert rows[(210, "CCC")]["Manual_Transcription"] == "hunkasabʰ"


def test_duplicate_rns_site_mapping_uncertainty_is_exhaustive():
    rns_rows = [row for row in MOD.load_cells() if row["Site_Key"].startswith("RNS_")]
    assert len(rns_rows) == 420
    assert all(row["Site_Assignment_Confidence"] == "medium" for row in rns_rows)
    assert all("duplicate source code RNS" in row["Uncertainty"] for row in rns_rows)
    with (ROOT / "unresolved_readings.tsv").open(encoding="utf-8", newline="") as handle:
        unresolved = list(csv.DictReader(handle, delimiter="\t"))
    assert len(unresolved) == 420
    assert {(int(row["Item"]), row["Site_Key"]) for row in unresolved} == {
        (item, site)
        for item in range(1, 211)
        for site in ("RNS_Sisaikhara", "RNS_Sisana")
    }
    assert {row["Issue_Type"] for row in unresolved} == {"site_mapping"}


def test_post_entry_legacy_reconciliation_is_exact_for_reviewed_targets():
    assert MOD.legacy_reconciliation(MOD.load_cells()) == {
        "manual_target_occurrences": 3560,
        "legacy_target_occurrences": 3548,
        "exact_occurrences": 2794,
        "manual_only_occurrences": 766,
        "legacy_only_occurrences": 754,
    }


def test_staging_accepts_complete_review():
    assert MOD.stage(MOD.load_cells()) is None


def test_staging_refuses_incomplete_or_unresolved_review():
    rows = MOD.load_cells()
    with pytest.raises(AssertionError, match="only 3359/3360 cells"):
        MOD.stage(rows[:-1])
    unresolved = [dict(row) for row in rows]
    unresolved[0]["Review_Status"] = "ambiguous"
    unresolved[0]["Manual_Transcription"] = ""
    unresolved[0]["Manual_Form_Count"] = "0"
    with pytest.raises(AssertionError, match="ambiguous or illegible cells remain"):
        MOD.stage(unresolved)
