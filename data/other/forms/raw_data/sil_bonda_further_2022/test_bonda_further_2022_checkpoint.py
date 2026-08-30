from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
MANUAL = HERE / "manual_chunks" / "items_001_005_hand_keyed.tsv"


def load_importer():
    path = HERE / "import_bonda_further_2022.py"
    spec = importlib.util.spec_from_file_location("bonda_further_importer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_topology_cannot_regress_to_fifteen_lists():
    registry = read_tsv(HERE / "list_registry.tsv")
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    topology = manifest["lexical_appendix"]
    assert len(registry) == topology["response_lists"] == 11
    assert topology["prompts"] == 210
    assert topology["conceptual_cells"] == 210 * 11 == 2310
    assert topology["target_lists"] == 3 and topology["target_cells"] == 630
    assert topology["comparison_lists"] == 8 and topology["comparison_cells"] == 1680
    assert topology["physical_pdf_pages"] == "15-47"
    checkpoint = manifest["manual_review_checkpoint"]
    assert checkpoint["completed_items"] == "1-210"
    assert checkpoint["remaining_items"] == "none" and checkpoint["remaining_cells"] == 0
    wrapped = {row["Site_Code"]: row["Printed_Label_2"] for row in registry if row["Printed_Label_2"]}
    assert wrapped == {"GUT": "Gadaba", "PAR": "Parenga Parja", "RON": "Desiya"}


def test_manual_ledger_is_ocr_blind_exhaustive_and_nfc():
    importer = load_importer()
    rows = importer.load_manual_cells()
    assert len(rows) == 210 * 11 == 2310
    assert {(row["Item"], row["Site_Code"]) for row in rows} == {
        (str(item), code) for item in range(1, 211) for code in importer.load_registry()
    }
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("15", "10") for row in rows if int(row["Item"]) <= 5)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("16", "11") for row in rows if 6 <= int(row["Item"]) <= 12)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("17", "12") for row in rows if 14 <= int(row["Item"]) <= 19)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("18", "13") for row in rows if 20 <= int(row["Item"]) <= 27)
    item13 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "13"}
    assert {item13[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}} == {("16", "11")}
    assert {item13[code] for code in {"GUT", "BIA", "PAR", "RON", "ODI"}} == {("17", "12")}
    item28 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "28"}
    assert {item28[code] for code in {"POD", "BON", "DUM", "KAD", "KEN"}} == {("18", "13")}
    assert {item28[code] for code in {"RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("19", "14")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("19", "14") for row in rows if 29 <= int(row["Item"]) <= 30)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("19", "14") for row in rows if 31 <= int(row["Item"]) <= 33)
    item34 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "34"}
    assert {item34[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT"}} == {("19", "14")}
    assert {item34[code] for code in {"BIA", "PAR", "RON", "ODI"}} == {("20", "15")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("20", "15") for row in rows if int(row["Item"]) == 35)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("20", "15") for row in rows if 36 <= int(row["Item"]) <= 40)
    item41 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "41"}
    assert {item41[code] for code in {"POD", "BON", "DUM", "KAD", "KEN"}} == {("20", "15")}
    assert {item41[code] for code in {"RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("21", "16")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("21", "16") for row in rows if 42 <= int(row["Item"]) <= 45)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("21", "16") for row in rows if int(row["Item"]) == 46)
    item47 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "47"}
    assert {item47[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA", "PAR"}} == {("21", "16")}
    assert {item47[code] for code in {"RON", "ODI"}} == {("22", "17")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("22", "17") for row in rows if 48 <= int(row["Item"]) <= 50)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("22", "17") for row in rows if 51 <= int(row["Item"]) <= 53)
    item54 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "54"}
    assert {item54[code] for code in {"POD", "BON", "DUM", "KAD"}} == {("22", "17")}
    assert {item54[code] for code in {"KEN", "RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("23", "18")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("23", "18") for row in rows if int(row["Item"]) == 55)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("23", "18") for row in rows if 56 <= int(row["Item"]) <= 59)
    item60 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "60"}
    assert {item60[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA", "PAR"}} == {("23", "18")}
    assert {item60[code] for code in {"RON", "ODI"}} == {("24", "19")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("24", "19") for row in rows if 61 <= int(row["Item"]) <= 65)
    item67 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "67"}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("24", "19") for row in rows if row["Item"] == "66")
    assert {item67[code] for code in {"POD", "BON", "DUM", "KAD"}} == {("24", "19")}
    assert {item67[code] for code in {"KEN", "RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("25", "20")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("25", "20") for row in rows if 68 <= int(row["Item"]) <= 70)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("25", "20") for row in rows if 71 <= int(row["Item"]) <= 73)
    item74 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "74"}
    assert {item74[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}} == {("25", "20")}
    assert {item74[code] for code in {"PAR", "RON", "ODI"}} == {("26", "21")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("26", "21") for row in rows if row["Item"] == "75")
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("26", "21") for row in rows if 76 <= int(row["Item"]) <= 80)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("27", "22") for row in rows if 81 <= int(row["Item"]) <= 85)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("27", "22") for row in rows if row["Item"] == "86")
    item87 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "87"}
    assert {item87[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}} == {("27", "22")}
    assert {item87[code] for code in {"PAR", "RON", "ODI"}} == {("28", "23")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("28", "23") for row in rows if 88 <= int(row["Item"]) <= 90)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("28", "23") for row in rows if 91 <= int(row["Item"]) <= 92)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("29", "24") for row in rows if 93 <= int(row["Item"]) <= 95)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("29", "24") for row in rows if 96 <= int(row["Item"]) <= 98)
    item99 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "99"}
    assert {item99[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}} == {("29", "24")}
    assert {item99[code] for code in {"PAR", "RON", "ODI"}} == {("30", "25")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("30", "25") for row in rows if row["Item"] == "100")
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("30", "25") for row in rows if 101 <= int(row["Item"]) <= 105)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("31", "26") for row in rows if 106 <= int(row["Item"]) <= 110)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("31", "26") for row in rows if row["Item"] == "111")
    item112 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "112"}
    assert {item112[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}} == {("31", "26")}
    assert {item112[code] for code in {"GUT", "BIA", "PAR", "RON", "ODI"}} == {("32", "27")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("32", "27") for row in rows if 113 <= int(row["Item"]) <= 115)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("32", "27") for row in rows if 116 <= int(row["Item"]) <= 118)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("33", "28") for row in rows if 119 <= int(row["Item"]) <= 120)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("33", "28") for row in rows if 121 <= int(row["Item"]) <= 124)
    item125 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "125"}
    assert {item125[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}} == {("33", "28")}
    assert {item125[code] for code in {"PAR", "RON", "ODI"}} == {("34", "29")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("34", "29") for row in rows if 126 <= int(row["Item"]) <= 130)
    item131 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "131"}
    assert {item131[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA", "PAR"}} == {("34", "29")}
    assert {item131[code] for code in {"RON", "ODI"}} == {("35", "30")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("35", "30") for row in rows if 132 <= int(row["Item"]) <= 135)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("35", "30") for row in rows if 136 <= int(row["Item"]) <= 137)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("36", "31") for row in rows if 138 <= int(row["Item"]) <= 140)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("36", "31") for row in rows if 141 <= int(row["Item"]) <= 143)
    item144 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "144"}
    assert {item144[code] for code in {"POD", "BON", "DUM", "KAD", "KEN"}} == {("36", "31")}
    assert {item144[code] for code in {"RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("37", "32")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("37", "32") for row in rows if row["Item"] == "145")
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("37", "32") for row in rows if 146 <= int(row["Item"]) <= 150)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("38", "33") for row in rows if 151 <= int(row["Item"]) <= 155)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("38", "33") for row in rows if row["Item"] == "156")
    item157 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "157"}
    assert {item157[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}} == {("38", "33")}
    assert {item157[code] for code in {"PAR", "RON", "ODI"}} == {("39", "34")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("39", "34") for row in rows if 158 <= int(row["Item"]) <= 160)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("39", "34") for row in rows if 161 <= int(row["Item"]) <= 163)
    item164 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "164"}
    assert {item164[code] for code in {"POD", "BON", "DUM", "KAD", "KEN"}} == {("39", "34")}
    assert {item164[code] for code in {"RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("40", "35")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("40", "35") for row in rows if row["Item"] == "165")
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("40", "35") for row in rows if 166 <= int(row["Item"]) <= 170)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("41", "36") for row in rows if 171 <= int(row["Item"]) <= 175)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("41", "36") for row in rows if row["Item"] == "176")
    item177 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "177"}
    assert {item177[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}} == {("41", "36")}
    assert {item177[code] for code in {"GUT", "BIA", "PAR", "RON", "ODI"}} == {("42", "37")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("42", "37") for row in rows if 178 <= int(row["Item"]) <= 180)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("42", "37") for row in rows if 181 <= int(row["Item"]) <= 182)
    item183 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "183"}
    assert {item183[code] for code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA", "PAR"}} == {("42", "37")}
    assert {item183[code] for code in {"RON", "ODI"}} == {("43", "38")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("43", "38") for row in rows if 184 <= int(row["Item"]) <= 185)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("43", "38") for row in rows if 186 <= int(row["Item"]) <= 189)
    item190 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "190"}
    assert {item190[code] for code in {"POD", "BON"}} == {("43", "38")}
    assert {item190[code] for code in {"DUM", "KAD", "KEN", "RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("44", "39")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("44", "39") for row in rows if 191 <= int(row["Item"]) <= 195)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("45", "40") for row in rows if 196 <= int(row["Item"]) <= 200)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("45", "40") for row in rows if row["Item"] == "201")
    item202 = {row["Site_Code"]: (row["PDF_Page"], row["Printed_Page"]) for row in rows if row["Item"] == "202"}
    assert {item202[code] for code in {"POD", "BON", "DUM", "KAD", "KEN"}} == {("45", "40")}
    assert {item202[code] for code in {"RAS", "GUT", "BIA", "PAR", "RON", "ODI"}} == {("46", "41")}
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("46", "41") for row in rows if 203 <= int(row["Item"]) <= 205)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("46", "41") for row in rows if 206 <= int(row["Item"]) <= 208)
    assert all((row["PDF_Page"], row["Printed_Page"]) == ("47", "42") for row in rows if 209 <= int(row["Item"]) <= 210)
    assert all(row["Reviewer_Declaration"] == importer.DECLARATION for row in rows)
    assert all(not any("ocr" in key.casefold() for key in row) for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())


def test_checkpoint_counts_and_scope_are_exact():
    importer = load_importer()
    forms, audit, counts = importer.build_checkpoint(importer.load_manual_cells())
    assert counts == {
        "reviewed_cells": 2310, "attested_cells": 2259,
        "source_blank_cells": 7, "excluded_cells": 44,
        "ambiguous_cells": 0, "illegible_cells": 0,
        "expanded_responses": 2394, "target_cells": 630, "target_forms": 644,
        "comparison_cells": 1680, "comparison_responses": 1750,
    }
    assert len(forms) == 644 and len(audit) == 2310
    assert Counter(row["Disposition"] for row in audit) == {
        "source-local target staging": 616,
        "audit-only: republished comparison list from JLSR 2022-004": 1643,
        "excluded: source prompt DISQUALIFIED": 44,
        "source blank: printed no entry": 7,
    }


def test_diplomatic_readings_variants_and_qualifier_are_preserved():
    rows = {}
    for path in sorted((HERE / "manual_chunks").glob("items_*_hand_keyed.tsv")):
        rows.update({(row["Item"], row["Site_Code"]): row for row in read_tsv(path)})
    assert rows[("2", "BON")]["Manual_Transcription"] == "bo:b'"
    assert rows[("3", "POD")]["Manual_Transcription"] == "ʊgʔbob' | ɭuibo"
    assert rows[("3", "POD")]["Similarity_Groups"] == "5|6"
    assert "(body hair)" in rows[("3", "POD")]["Source_Qualification"]
    assert rows[("3", "KAD")]["Manual_Transcription"] == "ʊgʔbo | ʊgʔbo"
    assert rows[("3", "KAD")]["Similarity_Groups"] == "2|5"
    assert rows[("3", "PAR")]["Manual_Transcription"] == "t̪ɪkuɪ"
    assert rows[("5", "RON")]["Manual_Transcription"] == "ɐ̃kɪ"
    assert rows[("6", "POD")]["Manual_Transcription"] == "lʊn̩t̪ʊr"
    assert rows[("6", "BIA")]["Manual_Transcription"] == "n̩lʊg̚"
    assert "combining left angle above" in rows[("6", "BIA")]["Source_Qualification"]
    assert rows[("7", "KEN")]["Manual_Transcription"] == "n̩tʃeʔmui"
    assert rows[("8", "KEN")]["Manual_Transcription"] == "t̪umo"
    assert rows[("8", "RON")]["Manual_Transcription"] == "ʈɔɳɖ"
    assert rows[("9", "RON")]["Manual_Transcription"] == "d̪ẽt̪"
    assert rows[("10", "BON")]["Manual_Transcription"] == "leʔjʌŋ"
    assert rows[("11", "POD")]["Review_Status"] == "excluded_disqualified"
    assert rows[("11", "POD")]["Manual_Transcription"] == ""
    assert rows[("12", "PAR")]["Manual_Transcription"] == "pot̪e | put̪e"
    assert rows[("14", "BON")]["Manual_Transcription"] == "sʊnʊkut̪i | sʊnʊkut̪i"
    assert "group-0" in rows[("14", "BON")]["Source_Qualification"]
    assert rows[("15", "ODI")]["Manual_Transcription"] == "toɭohato | papuli"
    assert rows[("16", "POD")]["Manual_Transcription"] == "nd̪roit̪i"
    assert rows[("16", "BIA")]["Manual_Transcription"] == "vʌɾvat̪i"
    assert rows[("17", "GUT")]["Manual_Transcription"] == "ɾʊmɪ | noq"
    assert rows[("17", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("17", "BIA")]["Manual_Transcription"] == "n̩tʃəit̪i"
    assert rows[("18", "RON")]["Manual_Transcription"] == "gəɾ"
    assert rows[("19", "DUM")]["Manual_Transcription"] == "ɪsa"
    assert rows[("19", "ODI")]["Manual_Transcription"] == "tʃaɾəmõ"
    assert rows[("20", "POD")]["Manual_Transcription"] == "sɪksəŋ"
    assert rows[("20", "RON")]["Manual_Transcription"] == "əɾ"
    assert rows[("21", "RAS")]["Manual_Transcription"] == "dʒimon | d̪ʊkd̪ukɪ"
    assert rows[("21", "RAS")]["Similarity_Groups"] == "1|5"
    assert rows[("21", "ODI")]["Manual_Transcription"] == "həɾud̪aio"
    assert rows[("22", "PAR")]["Manual_Transcription"] == "mɪjəŋ"
    assert rows[("23", "POD")]["Review_Status"] == "excluded_disqualified"
    assert rows[("24", "ODI")]["Manual_Transcription"] == ""
    assert rows[("25", "POD")]["Manual_Transcription"] == "ʊŋgəm"
    assert rows[("25", "KAD")]["Manual_Transcription"] == "uŋgəm"
    assert rows[("25", "RON")]["Manual_Transcription"] == "gẽ"
    assert rows[("26", "DUM")]["Manual_Transcription"] == "d̪iŋõ"
    assert rows[("26", "ODI")]["Manual_Transcription"] == "gɦoɾo"
    assert rows[("27", "POD")]["Manual_Transcription"] == "gʊd̪aŋbile"
    assert rows[("27", "RON")]["Manual_Transcription"] == "tʃɛɳɪ"
    assert rows[("28", "KAD")]["Manual_Transcription"] == "kəpat̪h"
    assert rows[("28", "PAR")]["Manual_Transcription"] == "nʌŋenu"
    assert rows[("29", "GUT")]["Manual_Transcription"] == "sʊø"
    assert rows[("30", "BON")]["Manual_Transcription"] == "sʊnʊʔ"
    assert rows[("30", "RON")]["Manual_Transcription"] == "bed̪ɳɪ"
    assert rows[("31", "POD")]["Manual_Transcription"] == "dʒʌn̪t̪a"
    assert rows[("31", "BON")]["Manual_Transcription"] == "bire | dʒʌn̪t̪a"
    assert rows[("31", "BON")]["Similarity_Groups"] == "6|1"
    assert "group-1 line with no response" in rows[("31", "BON")]["Source_Qualification"]
    assert rows[("31", "ODI")]["Manual_Transcription"] == "silɔ | kot̪t̪uni"
    assert rows[("32", "POD")]["Manual_Transcription"] == "dʒʌn̪t̪a"
    assert "group-0 line with no response" in rows[("32", "POD")]["Source_Qualification"]
    assert rows[("33", "POD")]["Manual_Transcription"] == "mʊtɭa"
    assert rows[("34", "BIA")]["Manual_Transcription"] == "suɾisəg˥"
    assert rows[("35", "KAD")]["Manual_Transcription"] == "kuɾad̪ɪ"
    assert rows[("35", "ODI")]["Manual_Transcription"] == "taŋgi:a"
    assert rows[("36", "DUM")]["Manual_Transcription"] == "gigei"
    assert rows[("36", "PAR")]["Manual_Transcription"] == "luʊeɾ"
    assert rows[("37", "POD")]["Manual_Transcription"] == "sut̪a"
    assert rows[("37", "ODI")]["Manual_Transcription"] == "su:t̪a"
    assert rows[("38", "BON")]["Manual_Transcription"] == "sʊdʒɪ"
    assert rows[("38", "ODI")]["Manual_Transcription"] == "sũn:tʃi"
    assert rows[("39", "KAD")]["Manual_Transcription"] == "m̩po"
    assert rows[("40", "PAR")]["Manual_Transcription"] == "mun̪d̪ɪ"
    assert rows[("40", "RON")]["Manual_Transcription"] == "mʊn̪d̪ɪ"
    assert rows[("41", "POD")]["Manual_Transcription"] == "siŋi"
    assert rows[("41", "GUT")]["Manual_Transcription"] == "siø"
    assert rows[("42", "PAR")]["Manual_Transcription"] == "ʌŋʌɪt̪ɛ"
    assert rows[("43", "KAD")]["Manual_Transcription"] == "akas | bed̪ol"
    assert rows[("43", "KAD")]["Similarity_Groups"] == "3|5"
    assert rows[("44", "RAS")]["Manual_Transcription"] == "momoɾt̪ɔ | kimit̪o"
    assert rows[("44", "ODI")]["Manual_Transcription"] == "nakʃat̪ɾa | t̪aɾa"
    assert rows[("45", "GUT")]["Manual_Transcription"] == "boɾsɛ | d̪ɛ"
    assert rows[("45", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("46", "BIA")]["Manual_Transcription"] == "n̪d̪ia"
    assert rows[("46", "RON")]["Manual_Transcription"] == "pəni"
    assert rows[("47", "POD")]["Manual_Transcription"] == "kɪn̪d̪a"
    assert rows[("47", "BIA")]["Manual_Transcription"] == "kɪn̪d̪iɛ"
    assert rows[("47", "RON")]["Manual_Transcription"] == "gɛɾ"
    assert rows[("48", "DUM")]["Manual_Transcription"] == "d̪aʔɲuɾgʊt̪a"
    assert rows[("48", "BIA")]["Manual_Transcription"] == "t̪ʊlet̪halodia"
    assert rows[("48", "ODI")]["Manual_Transcription"] == "megɦ:o"
    assert rows[("49", "POD")]["Manual_Transcription"] == "ʊŋleid̪a | sɪn̪t̪aɾ"
    assert rows[("49", "POD")]["Similarity_Groups"] == "6|7"
    assert rows[("49", "GUT")]["Manual_Transcription"] == "moglei | dʒɪtki"
    assert rows[("49", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("49", "RON")]["Manual_Transcription"] == "bɪdʒɪlɪ"
    assert rows[("50", "POD")]["Manual_Transcription"] == "gʊt̪ʊbʊɪ"
    assert rows[("50", "DUM")]["Manual_Transcription"] == "oŋt̪ɪbu"
    assert rows[("50", "RON")]["Manual_Transcription"] == "in̪d̪ɔɾd̪ʊn̪ʊ"
    assert rows[("50", "ODI")]["Manual_Transcription"] == "ind̪ɾod̪ənəsə"
    assert rows[("51", "POD")]["Manual_Transcription"] == "ʊɪd̪a"
    assert rows[("51", "BIA")]["Manual_Transcription"] == "ʊed̪ia"
    assert rows[("51", "ODI")]["Manual_Transcription"] == "dʒɦoɾaka"
    assert rows[("52", "BON")]["Manual_Transcription"] == "bʊɾe"
    assert rows[("52", "ODI")]["Manual_Transcription"] == "pət̪həɾə"
    assert rows[("53", "DUM")]["Manual_Transcription"] == "kʊɾʊŋ"
    assert rows[("53", "ODI")]["Manual_Transcription"] == "rast̪ɾa | bat̪o"
    assert rows[("53", "ODI")]["Similarity_Groups"] == "3|4"
    assert rows[("54", "GUT")]["Manual_Transcription"] == "bəli | bɪt̪ɪl"
    assert rows[("54", "GUT")]["Similarity_Groups"] == "1|3"
    assert rows[("54", "ODI")]["Manual_Transcription"] == "baɭi"
    assert rows[("55", "GUT")]["Manual_Transcription"] == "sʊøŋə:ol"
    assert rows[("55", "RON")]["Manual_Transcription"] == "dʒɔj"
    assert rows[("55", "ODI")]["Manual_Transcription"] == "nĩɑ"
    assert rows[("56", "DUM")]["Manual_Transcription"] == "mʊkʔsɪŋ"
    assert rows[("56", "ODI")]["Manual_Transcription"] == "d̪ɦuɑ̃"
    assert rows[("57", "KAD")]["Manual_Transcription"] == "ʊkʔsoŋ"
    assert rows[("57", "ODI")]["Manual_Transcription"] == "pɑ̃usə"
    assert rows[("58", "KAD")]["Manual_Transcription"] == "kʌd̪ot̪ʊbu | kʌd̪ot̪ʊbu"
    assert rows[("58", "KAD")]["Similarity_Groups"] == "4|5"
    assert rows[("58", "RAS")]["Manual_Transcription"] == "kʌsat̪ʊbu | kʌsat̪ʊbu"
    assert rows[("58", "RAS")]["Similarity_Groups"] == "2|5"
    assert rows[("58", "ODI")]["Manual_Transcription"] == "kad̪uə"
    assert rows[("59", "BIA")]["Manual_Transcription"] == "t̪hʊpʊɾlo"
    assert rows[("59", "ODI")]["Manual_Transcription"] == "d̪ɦuɭi"
    assert rows[("60", "GUT")]["Manual_Transcription"] == "sʊn:ɛ"
    assert rows[("60", "ODI")]["Manual_Transcription"] == "sun:a"
    assert rows[("61", "KAD")]["Manual_Transcription"] == "çemu"
    assert rows[("61", "RON")]["Manual_Transcription"] == "gɔtʃ"
    assert rows[("62", "PAR")]["Manual_Transcription"] == "ʊolɛ | ʊolɛ"
    assert rows[("62", "PAR")]["Similarity_Groups"] == "2|4"
    assert rows[("62", "ODI")]["Manual_Transcription"] == "pɔt̪ɔr"
    assert rows[("63", "POD")]["Manual_Transcription"] == "reɪgi"
    assert rows[("63", "BIA")]["Manual_Transcription"] == "n̪dʒrɛ"
    assert rows[("64", "PAR")]["Manual_Transcription"] == "ube | ʊolɛ"
    assert rows[("64", "PAR")]["Similarity_Groups"] == "3|5"
    assert rows[("64", "RON")]["Manual_Transcription"] == "kẽt̪e"
    assert rows[("65", "PAR")]["Manual_Transcription"] == "t̪ʌrbɛ"
    assert rows[("65", "ODI")]["Manual_Transcription"] == "phulo"
    assert rows[("66", "POD")]["Manual_Transcription"] == "po:lo"
    assert rows[("66", "BIA")]["Manual_Transcription"] == "tʃuɖe"
    assert rows[("67", "ODI")]["Manual_Transcription"] == "ɑmbo"
    assert rows[("68", "KEN")]["Manual_Transcription"] == "n̪dʒuʔnuɖa"
    assert rows[("68", "RON")]["Manual_Transcription"] == "kɔɖli"
    assert rows[("69", "PAR")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("69", "PAR")]["Manual_Transcription"] == ""
    assert rows[("69", "PAR")]["Similarity_Groups"] == "0"
    assert rows[("69", "ODI")]["Manual_Transcription"] == "gohomõ"
    assert all(rows[("70", code)]["Review_Status"] == "excluded_disqualified" for code in load_importer().load_registry())
    assert rows[("71", "DUM")]["Manual_Transcription"] == "ruŋkʊ"
    assert rows[("71", "GUT")]["Manual_Transcription"] == "rʊk:u"
    assert rows[("72", "ODI")]["Manual_Transcription"] == "aɭu"
    assert rows[("73", "GUT")]["Manual_Transcription"] == "ejom | beigon"
    assert rows[("73", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("73", "BIA")]["Manual_Transcription"] == "koɖẽhẽ"
    assert rows[("74", "KAD")]["Manual_Transcription"] == "tʃʌnɛ | tʃʌnɛ"
    assert rows[("74", "KAD")]["Similarity_Groups"] == "2|4"
    assert rows[("74", "ODI")]["Manual_Transcription"] == "tʃinboɖam | tʃinboɖam"
    assert rows[("74", "ODI")]["Similarity_Groups"] == "1|3"
    assert rows[("75", "RON")]["Manual_Transcription"] == "mɔritʃ"
    assert rows[("75", "ODI")]["Manual_Transcription"] == "mɔritʃə"
    assert rows[("76", "BIA")]["Manual_Transcription"] == "çiçia"
    assert rows[("76", "RON")]["Manual_Transcription"] == "ɔlɖɪ"
    assert rows[("77", "POD")]["Manual_Transcription"] == "t̪ʊlirʊsuɳo"
    assert rows[("77", "ODI")]["Manual_Transcription"] == "rəsuɳə"
    assert rows[("78", "GUT")]["Manual_Transcription"] == "pijɛdʒ | ʊl:i"
    assert rows[("78", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("79", "RON")]["Manual_Transcription"] == "phulkɔbi"
    assert rows[("80", "KAD")]["Manual_Transcription"] == "bedʒɪrɪ"
    assert rows[("80", "ODI")]["Manual_Transcription"] == "bilaʈi"
    assert rows[("81", "GUT")]["Manual_Transcription"] == "pʊɖekobi | bənɖegobi"
    assert rows[("81", "GUT")]["Similarity_Groups"] == "1|2"
    assert rows[("82", "GUT")]["Manual_Transcription"] == "soø:l"
    assert rows[("82", "BIA")]["Manual_Transcription"] == "n̩tʃu"
    assert rows[("83", "ODI")]["Manual_Transcription"] == "luŋə | nũno"
    assert rows[("83", "ODI")]["Similarity_Groups"] == "3|4"
    assert rows[("84", "GUT")]["Manual_Transcription"] == "sel:i"
    assert rows[("84", "PAR")]["Manual_Transcription"] == "ɕiɕi"
    assert rows[("85", "ODI")]["Manual_Transcription"] == "tʃərbi"
    assert rows[("86", "PAR")]["Manual_Transcription"] == "ʌju"
    assert rows[("87", "GUT")]["Manual_Transcription"] == "gis:iŋ"
    assert rows[("88", "POD")]["Manual_Transcription"] == "n̩t̪osiŋ"
    assert rows[("88", "ODI")]["Manual_Transcription"] == "oɳɖa"
    assert rows[("89", "POD")]["Manual_Transcription"] == "goɪt̪aŋ | dʒɔŋgoi"
    assert rows[("89", "POD")]["Similarity_Groups"] == "6|6"
    assert "`(female)` applies to the second response" in rows[("89", "POD")]["Source_Qualification"]
    assert rows[("89", "KAD")]["Manual_Transcription"] == "gɔiʔt̪ʌŋ | jɔŋgɔi | gɔiʔt̪ʌŋ"
    assert rows[("89", "KAD")]["Similarity_Groups"] == "3|5|6"
    assert rows[("90", "POD")]["Manual_Transcription"] == "bʊŋʈe | dʒɔŋbʊŋ"
    assert rows[("90", "POD")]["Similarity_Groups"] == "2|2"
    assert rows[("90", "KAD")]["Manual_Transcription"] == "bʊŋʈe | jɔŋbʊŋ"
    assert rows[("90", "KAD")]["Similarity_Groups"] == "2|5"
    assert rows[("91", "POD")]["Manual_Transcription"] == "d̪at̪ʊkʊi"
    assert rows[("91", "DUM")]["Manual_Transcription"] == "d̪at̪ikʊi"
    assert rows[("91", "KAD")]["Manual_Transcription"] == "d̪at̪ukui"
    assert rows[("92", "POD")]["Manual_Transcription"] == "ɖɔrʊŋ"
    assert rows[("92", "PAR")]["Manual_Transcription"] == "ɖʌru"
    assert rows[("93", "GUT")]["Manual_Transcription"] == "pʊɖɛ"
    assert rows[("93", "BIA")]["Manual_Transcription"] == "pɭa"
    assert rows[("93", "PAR")]["Manual_Transcription"] == "leŋdʒ"
    assert rows[("94", "GUT")]["Manual_Transcription"] == "gim:ɛ"
    assert rows[("94", "ODI")]["Manual_Transcription"] == "tʃheli"
    assert rows[("95", "DUM")]["Manual_Transcription"] == "gʊsʊ"
    assert rows[("95", "ODI")]["Manual_Transcription"] == "kukurɑ"
    assert rows[("96", "GUT")]["Manual_Transcription"] == "bʊɖboi"
    assert rows[("97", "BON")]["Manual_Transcription"] == "gisaʔ"
    assert rows[("97", "GUT")]["Manual_Transcription"] == "məkoɖ"
    assert rows[("98", "KAD")]["Manual_Transcription"] == "kirinjẽ"
    assert rows[("98", "GUT")]["Manual_Transcription"] == "bʊrsʊnɖi"
    assert rows[("99", "BIA")]["Manual_Transcription"] == "giɳaluo | buhi"
    assert rows[("99", "BIA")]["Similarity_Groups"] == "1|6"
    assert rows[("100", "GUT")]["Manual_Transcription"] == "kokoŋɖɛ | pɛt̪məkʊɖɪ"
    assert rows[("100", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("100", "ODI")]["Manual_Transcription"] == "buɖhiɑɳɪ"
    assert rows[("101", "GUT")]["Manual_Transcription"] == "imi | nev"
    assert rows[("101", "GUT")]["Similarity_Groups"] == "1|2"
    assert rows[("102", "POD")]["Manual_Transcription"] == "ŋgera | remo"
    assert rows[("102", "POD")]["Similarity_Groups"] == "6|6"
    assert "comma-separated" in rows[("102", "POD")]["Source_Qualification"]
    assert rows[("103", "PAR")]["Manual_Transcription"] == "guɳʈɔr | ʌmkur"
    assert rows[("103", "PAR")]["Similarity_Groups"] == "3|4"
    assert rows[("104", "RON")]["Manual_Transcription"] == "pɪlɛʈɔkɪ"
    assert rows[("105", "BIA")]["Manual_Transcription"] == "m̩ba"
    assert rows[("105", "PAR")]["Manual_Transcription"] == "ʌbɛ | ʌbɛ"
    assert rows[("105", "PAR")]["Similarity_Groups"] == "1|2"
    assert rows[("106", "POD")]["Manual_Transcription"] == "dʒoŋ"
    assert rows[("107", "KEN")]["Manual_Transcription"] == "mɳa maŋ"
    assert rows[("107", "ODI")]["Manual_Transcription"] == "nõnɑʔ"
    assert rows[("108", "KAD")]["Manual_Transcription"] == "biaŋ | meʔ"
    assert rows[("108", "KAD")]["Similarity_Groups"] == "6|6"
    assert rows[("108", "GUT")]["Manual_Transcription"] == "mijen bʊjɛŋ | mijenbɛi"
    assert rows[("108", "BIA")]["Manual_Transcription"] == "ɖhabõja | ɖhanepe"
    assert rows[("109", "ODI")]["Manual_Transcription"] == "nɑn:i | ɖiɖi"
    assert rows[("110", "GUT")]["Manual_Transcription"] == "mijent̪onen | mijen boini"
    assert rows[("110", "BIA")]["Manual_Transcription"] == "ɖhanet̪həɳɑ"
    assert rows[("111", "POD")]["Manual_Transcription"] == "õʔõ"
    assert rows[("111", "BIA")]["Manual_Transcription"] == "hũŋ"
    assert rows[("112", "BIA")]["Manual_Transcription"] == "selamboinehũ | selamboinehũ"
    assert rows[("112", "BIA")]["Similarity_Groups"] == "1|8"
    assert rows[("113", "POD")]["Manual_Transcription"] == "mpor"
    assert rows[("113", "KAD")]["Manual_Transcription"] == "m̩por"
    assert rows[("114", "GUT")]["Manual_Transcription"] == "kimoi | kʊmboi"
    assert rows[("114", "GUT")]["Similarity_Groups"] == "2|2"
    assert rows[("115", "BIA")]["Manual_Transcription"] == "ŋgirboʔo"
    assert rows[("115", "ODI")]["Manual_Transcription"] == "pilɑʔ | pu:o | pilɑʔ"
    assert rows[("115", "ODI")]["Similarity_Groups"] == "4|5|5"
    assert rows[("116", "KAD")]["Manual_Transcription"] == "ɖakui"
    assert rows[("117", "PAR")]["Manual_Transcription"] == "dʒɛɖɪŋ"
    assert rows[("118", "PAR")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("118", "PAR")]["Manual_Transcription"] == ""
    assert rows[("118", "PAR")]["Similarity_Groups"] == "0"
    assert rows[("119", "POD")]["Manual_Transcription"] == "ndʒur"
    assert rows[("119", "KAD")]["Manual_Transcription"] == "n̩dʒur"
    assert rows[("120", "GUT")]["Manual_Transcription"] == "simin | ɛɖʊbelɛ"
    assert rows[("120", "GUT")]["Similarity_Groups"] == "1|2"
    assert rows[("120", "ODI")]["Manual_Transcription"] == "məd̪hjan:ə"
    assert rows[("121", "BIA")]["Manual_Transcription"] == "ləmɖig˥"
    assert rows[("122", "POD")]["Manual_Transcription"] == "t̪ʊgola"
    assert rows[("123", "BIA")]["Manual_Transcription"] == "eiʔke | eiʔke"
    assert rows[("123", "BIA")]["Similarity_Groups"] == "1|5"
    assert rows[("124", "BIA")]["Manual_Transcription"] == "mɖʒoɖe"
    assert rows[("125", "KEN")]["Manual_Transcription"] == "muinsan̪t̪a"
    assert rows[("125", "ODI")]["Manual_Transcription"] == "səpt̪ahə"
    assert rows[("126", "DUM")]["Manual_Transcription"] == "arke | masek | mesek"
    assert rows[("126", "DUM")]["Similarity_Groups"] == "1|1|3"
    assert rows[("126", "GUT")]["Manual_Transcription"] == "mes | mes"
    assert rows[("126", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("127", "BIA")]["Manual_Transcription"] == "mimʊa"
    assert rows[("128", "RON")]["Manual_Transcription"] == "pʊrnɛ"
    assert rows[("128", "ODI")]["Manual_Transcription"] == "poruɳɑ"
    assert rows[("129", "RON")]["Manual_Transcription"] == "nũɛ̃"
    assert rows[("129", "ODI")]["Manual_Transcription"] == "nu:ɑ̃"
    assert rows[("130", "BIA")]["Manual_Transcription"] == "imanɖa | bɔl"
    assert rows[("130", "BIA")]["Similarity_Groups"] == "1|5"
    assert rows[("130", "ODI")]["Manual_Transcription"] == "bɦolo"
    assert rows[("131", "POD")]["Manual_Transcription"] == "bolʌra | olianɖra | bolʌra"
    assert rows[("131", "POD")]["Similarity_Groups"] == "6|8|8"
    assert rows[("131", "ODI")]["Manual_Transcription"] == "kɑrɑpo"
    assert rows[("132", "GUT")]["Manual_Transcription"] == "bʊgɖɛ"
    assert rows[("132", "BIA")]["Manual_Transcription"] == "brɔnɖe | lobonle"
    assert rows[("132", "BIA")]["Similarity_Groups"] == "1|7"
    assert rows[("133", "KAD")]["Manual_Transcription"] == "n̩dʒor"
    assert rows[("133", "BIA")]["Manual_Transcription"] == "n̩sʊar"
    assert rows[("134", "BIA")]["Manual_Transcription"] == "tʃilɛ"
    assert rows[("134", "PAR")]["Manual_Transcription"] == "ɖuŋkɛ"
    assert rows[("135", "KAD")]["Manual_Transcription"] == "ɖileboi | tʃorko baina"
    assert rows[("135", "KAD")]["Similarity_Groups"] == "1|4"
    assert rows[("135", "ODI")]["Manual_Transcription"] == "tsot̪ia"
    assert rows[("136", "GUT")]["Manual_Transcription"] == "sileinɖɛ | t̪orlo"
    assert rows[("136", "GUT")]["Similarity_Groups"] == "3|4"
    assert rows[("136", "RON")]["Manual_Transcription"] == "t̪ɔpɔt̪"
    assert rows[("137", "POD")]["Manual_Transcription"] == "sep'"
    assert rows[("137", "GUT")]["Manual_Transcription"] == "ruøo"
    assert rows[("138", "BIA")]["Manual_Transcription"] == "iŋtʃɔŋt̪i"
    assert rows[("139", "KAD")]["Manual_Transcription"] == "basɛ | basɛ"
    assert rows[("139", "KAD")]["Similarity_Groups"] == "2|6"
    assert rows[("139", "ODI")]["Manual_Transcription"] == "bɑ:mo"
    assert rows[("140", "POD")]["Manual_Transcription"] == "un̪t̪u"
    assert rows[("140", "ODI")]["Manual_Transcription"] == "pak:o"
    assert rows[("141", "POD")]["Manual_Transcription"] == "sʊlʊŋ"
    assert rows[("141", "PAR")]["Manual_Transcription"] == "ɖur"
    assert rows[("142", "POD")]["Manual_Transcription"] == "mʊna | bʊɖa | mʊna"
    assert rows[("142", "POD")]["Similarity_Groups"] == "1|1|5"
    assert rows[("142", "BIA")]["Manual_Transcription"] == "mɳa"
    assert rows[("143", "BIA")]["Manual_Transcription"] == "ɖãha"
    assert rows[("144", "KAD")]["Manual_Transcription"] == "leŋgɪ"
    assert rows[("144", "GUT")]["Manual_Transcription"] == "lɪgɪŋ | bodʒ"
    assert rows[("144", "GUT")]["Similarity_Groups"] == "1|2"
    assert rows[("144", "ODI")]["Manual_Transcription"] == "bɦaɾi"
    assert rows[("145", "RAS")]["Manual_Transcription"] == "ʊsɛs"
    assert rows[("145", "RON")]["Manual_Transcription"] == "usɛs"
    assert rows[("146", "BON")]["Manual_Transcription"] == "baʔbok"
    assert rows[("146", "GUT")]["Manual_Transcription"] == "t̪obnɛŋ"
    assert rows[("147", "KAD")]["Manual_Transcription"] == "ʌluŋ | dʒokt̪o"
    assert rows[("147", "KAD")]["Similarity_Groups"] == "1|2"
    assert rows[("147", "GUT")]["Manual_Transcription"] == "dʒot̪:o"
    assert rows[("147", "ODI")]["Manual_Transcription"] == "t̪əɭə"
    assert rows[("148", "DUM")]["Manual_Transcription"] == "tʊlʊi"
    assert rows[("148", "ODI")]["Manual_Transcription"] == "ɖhola"
    assert rows[("149", "PAR")]["Manual_Transcription"] == "ʌsɛɪ"
    assert rows[("149", "ODI")]["Manual_Transcription"] == "kolaʔ"
    assert rows[("150", "ODI")]["Manual_Transcription"] == "roŋgo | nali"
    assert rows[("150", "ODI")]["Similarity_Groups"] == "2|3"
    assert rows[("151", "POD")]["Manual_Transcription"] == "mʊjõ"
    assert rows[("151", "PAR")]["Manual_Transcription"] == "boɪ"
    assert rows[("152", "KAD")]["Manual_Transcription"] == "mbaʔar"
    assert rows[("152", "ODI")]["Manual_Transcription"] == "du:i"
    assert rows[("153", "KAD")]["Manual_Transcription"] == "t̪in̪t̪a"
    assert rows[("153", "BIA")]["Manual_Transcription"] == "n̪dʒi"
    assert rows[("154", "POD")]["Manual_Transcription"] == "ũʔũ"
    assert rows[("154", "ODI")]["Manual_Transcription"] == "tʃaɾi | tʃaɾgo"
    assert rows[("154", "ODI")]["Similarity_Groups"] == "2|2"
    assert rows[("155", "POD")]["Manual_Transcription"] == "past̪a"
    assert rows[("155", "ODI")]["Manual_Transcription"] == "pantʃə"
    assert rows[("156", "DUM")]["Manual_Transcription"] == "t̪ʔiri"
    assert rows[("156", "BIA")]["Manual_Transcription"] == "t̪ur"
    assert rows[("157", "POD")]["Manual_Transcription"] == "sat̪t̪a"
    assert rows[("157", "KAD")]["Manual_Transcription"] == "sɔt̪et̪a"
    assert rows[("157", "ODI")]["Manual_Transcription"] == "sat̪o"
    assert rows[("158", "BIA")]["Manual_Transcription"] == "t̪ʊma"
    assert rows[("159", "BIA")]["Manual_Transcription"] == "sʌŋt̪iŋ"
    assert rows[("159", "RON")]["Manual_Transcription"] == "nõ"
    assert rows[("160", "POD")]["Manual_Transcription"] == "ɖost̪a"
    assert rows[("160", "RON")]["Manual_Transcription"] == "ɖɔs"
    assert rows[("160", "ODI")]["Manual_Transcription"] == "ɖaso"
    assert rows[("161", "POD")]["Manual_Transcription"] == "egart̪a"
    assert rows[("161", "RON")]["Manual_Transcription"] == "ɛgɛr"
    assert rows[("162", "BIA")]["Manual_Transcription"] == "gɔmbar"
    assert rows[("163", "POD")]["Manual_Transcription"] == "kʊdije"
    assert rows[("163", "RAS")]["Manual_Transcription"] == "kuɖijet̪a"
    assert rows[("163", "ODI")]["Manual_Transcription"] == "koɾiˑe"
    assert rows[("164", "DUM")]["Manual_Transcription"] == "panskoɖi"
    assert rows[("164", "RON")]["Manual_Transcription"] == "pẽtʃkɔrɪ"
    assert rows[("165", "POD")]["Manual_Transcription"] == "dʒa"
    assert rows[("165", "BIA")]["Manual_Transcription"] == "dʒanɖe"
    assert rows[("165", "RON")]["Manual_Transcription"] == "kɔn̪t̪e"
    assert rows[("166", "GUT")]["Manual_Transcription"] == "mɛŋt̪e"
    assert rows[("166", "BIA")]["Manual_Transcription"] == "meʔ bare"
    assert rows[("167", "BIA")]["Manual_Transcription"] == "and̪i"
    assert rows[("167", "ODI")]["Manual_Transcription"] == "keuntare | kuade"
    assert rows[("167", "ODI")]["Similarity_Groups"] == "5|6"
    assert rows[("168", "POD")]["Manual_Transcription"] == "n̩d̪oi"
    assert rows[("168", "DUM")]["Manual_Transcription"] == "in̩d̪oja"
    assert rows[("169", "BIA")]["Manual_Transcription"] == "o:ʔd̪i"
    assert rows[("169", "RON")]["Manual_Transcription"] == "ket̪ət̪e"
    assert rows[("170", "DUM")]["Manual_Transcription"] == "mibai"
    assert rows[("170", "BIA")]["Manual_Transcription"] == "d̪eʔd̪irɔkɔm"
    assert rows[("170", "RON")]["Manual_Transcription"] == "ken̪t̪ət̪e"
    assert rows[("171", "POD")]["Manual_Transcription"] == "koʔn̪a"
    assert rows[("171", "PAR")]["Manual_Transcription"] == "ɪd̪ɪn"
    assert rows[("172", "BON")]["Manual_Transcription"] == "gʊtʊna | gʊtʊna"
    assert rows[("172", "BON")]["Similarity_Groups"] == "1|6"
    assert rows[("172", "DUM")]["Manual_Transcription"] == "gʊtʊna | gʊtʊna"
    assert rows[("173", "DUM")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("173", "DUM")]["Manual_Transcription"] == ""
    assert rows[("173", "BIA")]["Manual_Transcription"] == "khen̪iŋ"
    assert rows[("174", "DUM")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("174", "DUM")]["Manual_Transcription"] == ""
    assert rows[("174", "GUT")]["Manual_Transcription"] == "ʊn:u"
    assert rows[("175", "GUT")]["Manual_Transcription"] == "səm:ɛn | somɛn"
    assert rows[("175", "GUT")]["Similarity_Groups"] == "1|1"
    assert rows[("175", "PAR")]["Manual_Transcription"] == "ekepere"
    assert rows[("176", "GUT")]["Manual_Transcription"] == "bɪnbɪn"
    assert rows[("176", "ODI")]["Manual_Transcription"] == "bɦino bɦino | ələgɑ"
    assert rows[("176", "ODI")]["Similarity_Groups"] == "1|3"
    assert rows[("177", "BIA")]["Manual_Transcription"] == "sʌp:a"
    assert rows[("178", "KEN")]["Manual_Transcription"] == "puʔruga"
    assert rows[("179", "POD")]["Manual_Transcription"] == "uɪt̪ɔjo | una"
    assert rows[("179", "POD")]["Similarity_Groups"] == "1|1"
    assert "group-0" in rows[("179", "POD")]["Source_Qualification"]
    assert rows[("179", "BIA")]["Manual_Transcription"] == "ikud̪a | gond̪a"
    assert rows[("179", "BIA")]["Similarity_Groups"] == "1|2"
    assert rows[("180", "KAD")]["Manual_Transcription"] == "reʔt̪e | kʊb"
    assert rows[("180", "KAD")]["Similarity_Groups"] == "6|8"
    assert rows[("180", "GUT")]["Manual_Transcription"] == "kət̪:ijo"
    assert rows[("181", "BIA")]["Manual_Transcription"] == "t̪hʌnd̪e"
    assert rows[("182", "GUT")]["Manual_Transcription"] == "mɛisomo | som"
    assert rows[("182", "GUT")]["Similarity_Groups"] == "2|2"
    assert rows[("183", "KEN")]["Manual_Transcription"] == "ɔʔɔb"
    assert rows[("183", "RON")]["Manual_Transcription"] == "setʃɛblɛ"
    assert rows[("184", "POD")]["Manual_Transcription"] == "kʊd̪ʊgʊt̪a"
    assert rows[("184", "GUT")]["Manual_Transcription"] == "kʊd̪ʊgʊni | kʊd̪ʊgud̪ʊg:u"
    assert rows[("184", "GUT")]["Similarity_Groups"] == "2|2"
    assert rows[("185", "KAD")]["Manual_Transcription"] == "uː"
    assert rows[("185", "GUT")]["Manual_Transcription"] == "mɛid̪o | it̪unɪŋ"
    assert rows[("185", "BIA")]["Manual_Transcription"] == "uk̚ | me uke"
    assert rows[("185", "BIA")]["Similarity_Groups"] == "1|1"
    assert rows[("186", "BON")]["Manual_Transcription"] == "uʔd̪ad̪ʊsʊgut̪a"
    assert rows[("186", "GUT")]["Manual_Transcription"] == "mɛisosləgəigʊd̪ʊgu | sos"
    assert rows[("186", "GUT")]["Similarity_Groups"] == "3|3"
    assert rows[("187", "GUT")]["Manual_Transcription"] == "d̪ud̪i | mɛid̪ud̪igu"
    assert rows[("187", "BIA")]["Manual_Transcription"] == "d̪ulaik | d̪ulaige"
    assert rows[("188", "RAS")]["Manual_Transcription"] == "dʒokt̪od̪rɪt̪a"
    assert rows[("188", "RON")]["Manual_Transcription"] == "sed̪ul:ɛ"
    assert rows[("189", "GUT")]["Manual_Transcription"] == "leisɛ | mɛileigɪ"
    assert rows[("189", "BIA")]["Manual_Transcription"] == "kola | koke"
    assert rows[("190", "GUT")]["Manual_Transcription"] == "ɪn̪d̪e | mɛibed̪o"
    assert rows[("190", "GUT")]["Similarity_Groups"] == "1|2"
    assert rows[("190", "ODI")]["Manual_Transcription"] == "d̪eba"
    assert rows[("191", "GUT")]["Manual_Transcription"] == "ləgəigʊni | mɛigɛbo"
    assert rows[("191", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("191", "BIA")]["Manual_Transcription"] == "d̪uʋad̪iŋke | tʃigmuaga"
    assert rows[("192", "GUT")]["Manual_Transcription"] == "goisɛ | mɛigoigɪ | oɾ goigu"
    assert rows[("192", "GUT")]["Similarity_Groups"] == "1|1|1"
    assert rows[("192", "ODI")]["Manual_Transcription"] == "moɾonõ"
    assert rows[("193", "GUT")]["Manual_Transcription"] == "mɛibʊo | bʊq"
    assert rows[("193", "BIA")]["Manual_Transcription"] == "aboge | me bagoige"
    assert rows[("194", "POD")]["Manual_Transcription"] == "ʋalo"
    assert rows[("194", "GUT")]["Manual_Transcription"] == "mɛiʋd̪eigʊ | ʊd̪ei"
    assert rows[("195", "GUT")]["Manual_Transcription"] == "əŋsʊŋ:ɛ | mɛiəŋsʊŋgu"
    assert rows[("195", "ODI")]["Manual_Transcription"] == "tʃalibɑʔ"
    assert rows[("196", "POD")]["Manual_Transcription"] == "ur"
    assert rows[("196", "GUT")]["Manual_Transcription"] == "mɛid̪ʊŋgu | d̪ʊe"
    assert rows[("196", "GUT")]["Similarity_Groups"] == "2|3"
    assert rows[("196", "RON")]["Manual_Transcription"] == "se pelɛjlɛ"
    assert rows[("197", "GUT")]["Manual_Transcription"] == "mɛiʋidʒi | jɛ"
    assert rows[("197", "BIA")]["Manual_Transcription"] == "veglɑ"
    assert rows[("198", "KAD")]["Manual_Transcription"] == "lo:"
    assert rows[("198", "GUT")]["Manual_Transcription"] == "mɛipiŋgi | olo"
    assert rows[("198", "PAR")]["Manual_Transcription"] == "bɛɪ"
    assert rows[("198", "ODI")]["Manual_Transcription"] == "ɑ:so"
    assert rows[("199", "POD")]["Manual_Transcription"] == "sũ"
    assert rows[("199", "GUT")]["Manual_Transcription"] == "mɛisun:o | sun"
    assert rows[("199", "ODI")]["Manual_Transcription"] == "kɔhilɑ | kuhɑ"
    assert rows[("199", "ODI")]["Similarity_Groups"] == "4|4"
    assert rows[("200", "POD")]["Manual_Transcription"] == "õŋ"
    assert rows[("200", "GUT")]["Manual_Transcription"] == "mɛioʔoø | oŋ"
    assert rows[("200", "BIA")]["Manual_Transcription"] == "nahot̪e"
    assert rows[("200", "RON")]["Manual_Transcription"] == "se sʊnlɛ"
    assert rows[("201", "KAD")]["Manual_Transcription"] == "dʒu:"
    assert rows[("201", "GUT")]["Manual_Transcription"] == "dʒu | mɛidʒuvo"
    assert rows[("201", "GUT")]["Similarity_Groups"] == "2|2"
    assert rows[("201", "RON")]["Manual_Transcription"] == "se d̪eklɛ"
    assert rows[("202", "POD")]["Manual_Transcription"] == "niŋ"
    assert rows[("202", "BIA")]["Manual_Transcription"] == "naiŋ"
    assert rows[("202", "RON")]["Manual_Transcription"] == "mũɪ"
    assert rows[("203", "GUT")]["Manual_Transcription"] == "nom"
    assert rows[("203", "RON")]["Manual_Transcription"] == "t̪uɪ"
    assert rows[("204", "KEN")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("204", "KEN")]["Manual_Transcription"] == ""
    assert rows[("204", "BIA")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("204", "ODI")]["Manual_Transcription"] == "ɑponõ"
    assert rows[("205", "POD")]["Manual_Transcription"] == "mai"
    assert rows[("205", "GUT")]["Manual_Transcription"] == "mɛi"
    assert rows[("205", "RON")]["Manual_Transcription"] == "se"
    assert rows[("206", "POD")]["Manual_Transcription"] == "mai"
    assert rows[("206", "GUT")]["Manual_Transcription"] == "mɛi"
    assert rows[("206", "BIA")]["Manual_Transcription"] == "merɑ"
    assert rows[("207", "GUT")]["Manual_Transcription"] == "nɛinen"
    assert rows[("207", "PAR")]["Manual_Transcription"] == "bilɛŋ"
    assert rows[("207", "ODI")]["Manual_Transcription"] == "ɑme | ɑmpe"
    assert rows[("207", "ODI")]["Similarity_Groups"] == "5|5"
    assert rows[("208", "BIA")]["Manual_Transcription"] == "ok:en remo"
    assert rows[("208", "PAR")]["Review_Status"] == "source_blank_no_entry"
    assert rows[("208", "PAR")]["Manual_Transcription"] == ""
    assert rows[("208", "ODI")]["Manual_Transcription"] == "ɑme | ɑmpe"
    assert rows[("209", "KAD")]["Manual_Transcription"] == "pele"
    assert rows[("209", "GUT")]["Manual_Transcription"] == "pɛn"
    assert rows[("209", "PAR")]["Manual_Transcription"] == "mɛŋdʒɪ"
    assert rows[("209", "RON")]["Manual_Transcription"] == "t̪ume"
    assert rows[("210", "POD")]["Manual_Transcription"] == "meʔje"
    assert rows[("210", "KAD")]["Manual_Transcription"] == "maʔɪle"
    assert rows[("210", "KEN")]["Manual_Transcription"] == "maʔɛ:"
    assert rows[("210", "BIA")]["Manual_Transcription"] == "mehɲ"
    assert rows[("210", "PAR")]["Manual_Transcription"] == "ʌd̪iŋmɔɪ"
    assert rows[("210", "ODI")]["Manual_Transcription"] == "se mɑn:e"


def test_only_targets_are_staged_and_dum_is_replacement_target():
    registry = {row["Site_Code"]: row for row in read_tsv(HERE / "list_registry.tsv")}
    assert {code for code, row in registry.items() if row["Install"] == "yes"} == {"POD", "BON", "DUM"}
    assert registry["DUM"]["Scope"] == "checked_replacement_target"
    assert registry["DUM"]["Dialect_ID"] == "sil-bonda-didayi-1997-dumripada-u-bonda"
    assert all(row["Install"] == "no" for code, row in registry.items() if code not in {"POD", "BON", "DUM"})
    forms = list(csv.reader((HERE / "checkpoint_forms.csv").open(encoding="utf-8", newline="")))
    assert len(forms) == 644
    assert all(row[0] == "re" and row[11] == "" for row in forms)
    assert all("Kadamguda" not in row[14] and "Cuttack" not in row[14] for row in forms)


def test_comparison_reconciliation_is_separate_and_exact():
    rows = read_tsv(HERE / "comparison_reconciliation.tsv")
    assert len(rows) == 1680
    assert Counter(row["Match_Status"] for row in rows) == {
        "exact-diplomatic-match": 580,
        "length-mark-rendering-equivalent": 42,
        "repeated-current-response": 4,
        "different-current-printing": 1054,
    }
    differences = {(row["Item"], row["Site_Code"]) for row in rows if row["Match_Status"] == "different-current-printing"}
    assert {("1", "PAR"), ("1", "RON"), ("2", "PAR"), ("3", "PAR"), ("3", "ODI"), ("5", "RON")} <= differences
    assert {("6", "BIA"), ("7", "KEN"), ("8", "RON"), ("10", "GUT")} <= {
        (row["Item"], row["Site_Code"]) for row in rows if row["Match_Status"] == "exact-diplomatic-match"
    }


def test_dumripada_checked_replacement_is_exhaustively_reconciled():
    rows = read_tsv(HERE / "dumripada_replacement_reconciliation.tsv")
    assert len(rows) == 210
    assert {row["Item"] for row in rows} == {str(item) for item in range(1, 211)}
    assert {row["Current_Site_Code"] for row in rows} == {"DUM"}
    assert {row["Prior_Site_Code"] for row in rows} == {"DUM"}
    assert {row["Integration_Disposition"] for row in rows} == {
        "checked-2002-list-current; 1997-list-superseded-audit-only"
    }
    assert Counter(row["Match_Status"] for row in rows) == {
        "both-unattested": 5,
        "current-attested-prior-unattested": 4,
        "current-source_blank_no_entry-prior-attested": 1,
        "different-checked-current-printing": 165,
        "exact-diplomatic-match": 34,
        "length-mark-rendering-equivalent": 1,
    }
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    replacement = manifest["dumripada_replacement_reconciliation"]
    assert replacement["reviewed_replacement_cells"] == 210
    assert replacement["status"] == dict(Counter(row["Match_Status"] for row in rows))


def test_profile_covers_every_staged_form():
    importer = load_importer()
    profile = importer.load_profile()
    assert len(profile) == 50
    forms = list(csv.reader((HERE / "checkpoint_forms.csv").open(encoding="utf-8", newline="")))
    assert forms
    assert all(importer.convert(row[2], profile) for row in forms)


def test_seeded_twenty_cell_source_to_audit_sample_is_exact():
    sample_keys = [
        ("1", "POD"), ("11", "RON"), ("21", "RAS"), ("32", "BON"),
        ("43", "DUM"), ("54", "GUT"), ("65", "BIA"), ("76", "PAR"),
        ("87", "ODI"), ("98", "KAD"), ("109", "KEN"), ("120", "GUT"),
        ("131", "RON"), ("142", "POD"), ("153", "BON"), ("164", "BIA"),
        ("175", "PAR"), ("186", "GUT"), ("197", "ODI"), ("208", "PAR"),
    ]
    manual = {}
    for path in sorted((HERE / "manual_chunks").glob("items_*_hand_keyed.tsv")):
        manual.update({(row["Item"], row["Site_Code"]): row for row in read_tsv(path)})
    audit = {(row["Item"], row["Site_Code"]): row for row in read_tsv(HERE / "checkpoint_audit.tsv")}
    preserved = {
        "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
        "Column", "Manual_Transcription", "Similarity_Groups",
        "Source_Qualification", "Review_Status", "Confidence", "Uncertainty",
        "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
    }
    assert len(sample_keys) == len(set(sample_keys)) == 20
    assert all({field: audit[key][field] for field in preserved} == manual[key] for key in sample_keys)


def test_manifest_hashes_reproduce_and_unresolved_is_header_only():
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]["manual_ledgers"]:
        assert digest(HERE / item["path"]) == item["sha256"]
    for key, item in manifest["artifacts"].items():
        if key == "manual_ledgers":
            continue
        assert digest(HERE / item["path"]) == item["sha256"]
    unresolved = (HERE / "unresolved_readings.tsv").read_text(encoding="utf-8").splitlines()
    assert len(unresolved) == 1
    assert manifest["manual_review_checkpoint"]["unresolved_transcriptions"] == []


def test_shared_integration_files_match_the_frozen_package():
    data_root = HERE.parents[4]
    installed = data_root / "data" / "other" / "forms" / "20260829-sil-bonda-further.csv"
    shared_profile = data_root / "conversion" / "sil-bonda-further.txt"
    assert digest(installed) == digest(HERE / "checkpoint_forms.csv")
    assert len(list(csv.reader(installed.open(encoding="utf-8", newline="")))) == 644
    assert shared_profile.read_text(encoding="utf-8") == (HERE / "conversion_profile.tsv").read_text(encoding="utf-8")

    dialects = (data_root / "cldf" / "dialects.csv").read_text(encoding="utf-8")
    assert "sil-bonda-further-2002-podeiguda-u-bonda" in dialects
    assert "sil-bonda-further-2002-bondapada-u-bonda" in dialects
    sources = (data_root / "cldf" / "sources.bib").read_text(encoding="utf-8")
    assert "@techreport{mathew2022bonda-further," in sources
    builder = (data_root / "make_cldf.py").read_text(encoding="utf-8")
    assert 'source_key == "mathew2022bonda-further"' in builder
    assert 'row_ipa = "sil-bonda-further"' in builder
    assert 'source_key == "mathew-chamberlain2022bonda-didayi"' in builder
    assert "sil-bonda-didayi-1997-dumripada-u-bonda" in builder


def test_guard_rejects_any_ocr_bearing_ledger(tmp_path: Path):
    importer = load_importer()
    with MANUAL.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or []) + ["OCR_Evidence"]
    bad = tmp_path / "ocr_bearing.tsv"
    with bad.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "OCR_Evidence": "forbidden"})
    with pytest.raises(AssertionError, match="unexpected ledger schema"):
        importer.load_manual_cells(bad)
