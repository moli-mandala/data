import csv
import importlib.util
import unicodedata
from pathlib import Path

import pytest
from segments import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_bhumij_2015"
LEDGER_001_004 = PACKAGE / "manual_chunks/items_001_004_hand_keyed.tsv"
LEDGER_005_009 = PACKAGE / "manual_chunks/items_005_009_hand_keyed.tsv"
LEDGER_010_014 = PACKAGE / "manual_chunks/items_010_014_hand_keyed.tsv"
LEDGER_015_019 = PACKAGE / "manual_chunks/items_015_019_hand_keyed.tsv"
LEDGER_020_024 = PACKAGE / "manual_chunks/items_020_024_hand_keyed.tsv"
LEDGER_025_029 = PACKAGE / "manual_chunks/items_025_029_hand_keyed.tsv"
LEDGER_030_034 = PACKAGE / "manual_chunks/items_030_034_hand_keyed.tsv"
LEDGER_035_039 = PACKAGE / "manual_chunks/items_035_039_hand_keyed.tsv"
LEDGER_040_044 = PACKAGE / "manual_chunks/items_040_044_hand_keyed.tsv"
LEDGER_045_049 = PACKAGE / "manual_chunks/items_045_049_hand_keyed.tsv"
LEDGER_050_054 = PACKAGE / "manual_chunks/items_050_054_hand_keyed.tsv"
LEDGER_055_059 = PACKAGE / "manual_chunks/items_055_059_hand_keyed.tsv"
LEDGER_060_064 = PACKAGE / "manual_chunks/items_060_064_hand_keyed.tsv"
LEDGER_065_069 = PACKAGE / "manual_chunks/items_065_069_hand_keyed.tsv"
LEDGER_070_074 = PACKAGE / "manual_chunks/items_070_074_hand_keyed.tsv"
LEDGER_075_079 = PACKAGE / "manual_chunks/items_075_079_hand_keyed.tsv"
LEDGER_080_084 = PACKAGE / "manual_chunks/items_080_084_hand_keyed.tsv"
LEDGER_085_089 = PACKAGE / "manual_chunks/items_085_089_hand_keyed.tsv"
LEDGER_090_094 = PACKAGE / "manual_chunks/items_090_094_hand_keyed.tsv"
LEDGER_095_099 = PACKAGE / "manual_chunks/items_095_099_hand_keyed.tsv"
LEDGER_100_104 = PACKAGE / "manual_chunks/items_100_104_hand_keyed.tsv"
LEDGER_105_109 = PACKAGE / "manual_chunks/items_105_109_hand_keyed.tsv"
LEDGER_110_114 = PACKAGE / "manual_chunks/items_110_114_hand_keyed.tsv"
LEDGER_115_119 = PACKAGE / "manual_chunks/items_115_119_hand_keyed.tsv"
LEDGER_120_124 = PACKAGE / "manual_chunks/items_120_124_hand_keyed.tsv"
LEDGER_125_129 = PACKAGE / "manual_chunks/items_125_129_hand_keyed.tsv"
LEDGER_130_134 = PACKAGE / "manual_chunks/items_130_134_hand_keyed.tsv"
LEDGER_135_139 = PACKAGE / "manual_chunks/items_135_139_hand_keyed.tsv"
LEDGER_140_144 = PACKAGE / "manual_chunks/items_140_144_hand_keyed.tsv"
LEDGER_145_149 = PACKAGE / "manual_chunks/items_145_149_hand_keyed.tsv"
LEDGER_150_154 = PACKAGE / "manual_chunks/items_150_154_hand_keyed.tsv"
LEDGER_155_159 = PACKAGE / "manual_chunks/items_155_159_hand_keyed.tsv"
LEDGER_160_164 = PACKAGE / "manual_chunks/items_160_164_hand_keyed.tsv"
LEDGER_165_169 = PACKAGE / "manual_chunks/items_165_169_hand_keyed.tsv"
LEDGER_170_174 = PACKAGE / "manual_chunks/items_170_174_hand_keyed.tsv"
LEDGER_175_179 = PACKAGE / "manual_chunks/items_175_179_hand_keyed.tsv"
LEDGER_180_181 = PACKAGE / "manual_chunks/items_180_181_hand_keyed.tsv"
LEDGER_182 = PACKAGE / "manual_chunks/items_182_182_hand_keyed.tsv"
LEDGER_183 = PACKAGE / "manual_chunks/items_183_183_hand_keyed.tsv"
LEDGER_184 = PACKAGE / "manual_chunks/items_184_184_hand_keyed.tsv"
LEDGER_185 = PACKAGE / "manual_chunks/items_185_185_hand_keyed.tsv"
LEDGER_186 = PACKAGE / "manual_chunks/items_186_186_hand_keyed.tsv"
LEDGER_187 = PACKAGE / "manual_chunks/items_187_187_hand_keyed.tsv"
LEDGER_188 = PACKAGE / "manual_chunks/items_188_188_hand_keyed.tsv"
LEDGER_189 = PACKAGE / "manual_chunks/items_189_189_hand_keyed.tsv"
LEDGER_190 = PACKAGE / "manual_chunks/items_190_190_hand_keyed.tsv"
LEDGER_191 = PACKAGE / "manual_chunks/items_191_191_hand_keyed.tsv"
LEDGER_192 = PACKAGE / "manual_chunks/items_192_192_hand_keyed.tsv"
LEDGER_193 = PACKAGE / "manual_chunks/items_193_193_hand_keyed.tsv"
LEDGER_194 = PACKAGE / "manual_chunks/items_194_194_hand_keyed.tsv"
LEDGER_195 = PACKAGE / "manual_chunks/items_195_195_hand_keyed.tsv"
LEDGER_196 = PACKAGE / "manual_chunks/items_196_196_hand_keyed.tsv"
LEDGER_197 = PACKAGE / "manual_chunks/items_197_197_hand_keyed.tsv"
LEDGER_198 = PACKAGE / "manual_chunks/items_198_198_hand_keyed.tsv"
LEDGER_199 = PACKAGE / "manual_chunks/items_199_199_hand_keyed.tsv"
LEDGER_200 = PACKAGE / "manual_chunks/items_200_200_hand_keyed.tsv"
LEDGER_201 = PACKAGE / "manual_chunks/items_201_201_hand_keyed.tsv"
LEDGER_202 = PACKAGE / "manual_chunks/items_202_202_hand_keyed.tsv"
LEDGER_203 = PACKAGE / "manual_chunks/items_203_203_hand_keyed.tsv"
LEDGER_204_208 = PACKAGE / "manual_chunks/items_204_208_hand_keyed.tsv"
LEDGER_209_210 = PACKAGE / "manual_chunks/items_209_210_hand_keyed.tsv"
LEDGERS = [
    LEDGER_001_004, LEDGER_005_009, LEDGER_010_014, LEDGER_015_019,
    LEDGER_020_024, LEDGER_025_029, LEDGER_030_034, LEDGER_035_039,
    LEDGER_040_044, LEDGER_045_049, LEDGER_050_054, LEDGER_055_059,
    LEDGER_060_064, LEDGER_065_069, LEDGER_070_074, LEDGER_075_079,
    LEDGER_080_084, LEDGER_085_089, LEDGER_090_094, LEDGER_095_099,
    LEDGER_100_104,
    LEDGER_105_109,
    LEDGER_110_114,
    LEDGER_115_119,
    LEDGER_120_124,
    LEDGER_125_129,
    LEDGER_130_134,
    LEDGER_135_139,
    LEDGER_140_144,
    LEDGER_145_149,
    LEDGER_150_154,
    LEDGER_155_159,
    LEDGER_160_164,
    LEDGER_165_169,
    LEDGER_170_174,
    LEDGER_175_179,
    LEDGER_180_181,
    LEDGER_182,
    LEDGER_183,
    LEDGER_184,
    LEDGER_185,
    LEDGER_186,
    LEDGER_187,
    LEDGER_188,
    LEDGER_189,
    LEDGER_190,
    LEDGER_191,
    LEDGER_192,
    LEDGER_193,
    LEDGER_194,
    LEDGER_195,
    LEDGER_196,
    LEDGER_197,
    LEDGER_198,
    LEDGER_199,
    LEDGER_200,
    LEDGER_201,
    LEDGER_202,
    LEDGER_203,
    LEDGER_204_208,
    LEDGER_209_210,
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("bhumij_guard", PACKAGE / "import_bhumij_2015.py")
overlap = load_module(
    "bhumij_overlap", PACKAGE / "build_overlap_reconciliation.py"
)


def test_first_chunk_is_complete_and_manual():
    rows = guard.load_manual_cells(LEDGER_001_004)
    assert len(rows) == 4 * 18 == 72
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 72
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())


def test_accounting_targets_controls_and_stage_guard():
    rows = guard.load_manual_ledgers(LEDGERS)
    assert len(rows) == 210 * 18 == 3780
    assert sum(row["Review_Status"] == "attested" for row in rows) == 3690
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 90
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 3876
    assert len(guard.stage_target_forms(rows)) == 2100
    assert sum(row["Target"] == "no" for row in rows) == 1680
    guard.require_full_review(rows)


def test_guard_rejects_ocr_field_or_missing_declaration(tmp_path):
    rows = list(csv.DictReader(LEDGER_001_004.open(encoding="utf-8"), delimiter="\t"))
    bad_ocr = tmp_path / "bad_ocr.tsv"
    fieldnames = list(rows[0]) + ["OCR_Evidence"]
    with bad_ocr.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "not admissible"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_ocr)

    bad_declaration = tmp_path / "bad_declaration.tsv"
    rows[0]["Reviewer_Declaration"] = ""
    with bad_declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_declaration)


def test_second_chunk_coordinates_variants_and_difficult_symbols():
    rows = guard.load_manual_cells(LEDGER_005_009)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("5", "BAI")]["PDF_Page"] == "34"
    assert by_key[("5", "BAI")]["Column"] == "right"
    assert by_key[("5", "MUN")]["PDF_Page"] == "35"
    assert by_key[("5", "MUN")]["Column"] == "left"
    assert by_key[("5", "LAD")]["Manual_Transcription"] == "metʔn̩"
    assert by_key[("5", "UDA")]["Manual_Transcription"] == "meʔt̪"
    assert by_key[("6", "BAI")]["Manual_Transcription"] == "lut̪uɾ"
    assert by_key[("8", "MDI")]["Manual_Transcription"] == "motʃɑ | thotnɑ"
    assert by_key[("8", "MDI")]["Source_Cognate_Labels"] == "1 | 4"
    assert by_key[("9", "LAD")]["Manual_Transcription"] == "d̪ɑʔtɑ"
    assert by_key[("9", "ORI")]["Manual_Transcription"] == "d̪ɑnt̪o"


def test_third_chunk_blanks_coordinates_and_multiform_cells():
    rows = guard.load_manual_cells(LEDGER_010_014)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("10", "BAI")]["PDF_Page"] == "35"
    assert by_key[("10", "MCH")]["PDF_Page"] == "36"
    assert by_key[("10", "MDI")]["Manual_Transcription"] == "ɑlɑŋ | leʔe"
    assert by_key[("11", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("11", "MUN")]["Review_Status"] == "source_blank"
    assert by_key[("11", "SNA")]["Review_Status"] == "source_blank"
    assert by_key[("11", "SDI")]["Manual_Transcription"] == "koɾɑm | nunu"
    assert by_key[("12", "SDI")]["Manual_Transcription"] == "lɑʔe | dodʒok"
    assert by_key[("13", "SDI")]["Manual_Transcription"] == "t̪i | sopo"
    assert by_key[("14", "DIG")]["Manual_Transcription"] == "uk | ukɑ"


def test_fourth_chunk_page_splits_blank_and_multiform_cells():
    rows = guard.load_manual_cells(LEDGER_015_019)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("15", "BAI")]["PDF_Page"] == "36"
    assert by_key[("15", "UDA")]["PDF_Page"] == "37"
    assert by_key[("15", "ORI")]["Manual_Transcription"] == "toɭohɑto | pɑpuli"
    assert by_key[("16", "LAD")]["Manual_Transcription"] == "ɖɑd̪o"
    assert by_key[("17", "SDI")]["Manual_Transcription"] == "t̪i ɾɑmɑ"
    assert by_key[("18", "BAI")]["Column"] == "left"
    assert by_key[("18", "CHA")]["Column"] == "right"
    assert by_key[("19", "MDI")]["Manual_Transcription"] == "hɑɾtɑ | ũɾ"
    assert by_key[("19", "HDI")]["Review_Status"] == "source_blank"


def test_fifth_chunk_page_splits_blanks_and_multiform_cells():
    rows = guard.load_manual_cells(LEDGER_020_024)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("20", "BAI")]["PDF_Page"] == "37"
    assert by_key[("20", "MJH")]["PDF_Page"] == "38"
    assert by_key[("20", "HDI")]["Review_Status"] == "source_blank"
    assert by_key[("21", "MDI")]["Manual_Transcription"] == "dʒi | bukɑ"
    assert by_key[("21", "SDI")]["Manual_Transcription"] == "boko | ontoɾ"
    assert by_key[("22", "MDI")]["Manual_Transcription"] == "mɑjom | ɾokot"
    assert by_key[("23", "DUM")]["Column"] == "left"
    assert by_key[("23", "LAD")]["Column"] == "right"
    assert by_key[("23", "POD")]["Review_Status"] == "source_blank"
    assert by_key[("24", "SDI")]["Manual_Transcription"] == "dʒidʒɑ | itʃʔ"


def test_sixth_chunk_page_splits_and_multiform_cells():
    rows = guard.load_manual_cells(LEDGER_025_029)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("25", "BAI")]["PDF_Page"] == "38"
    assert by_key[("25", "MJH")]["PDF_Page"] == "39"
    assert by_key[("25", "MDI")]["Manual_Transcription"] == "hɑt̪u | ɖi"
    assert by_key[("26", "MJH")]["Manual_Transcription"] == "ɔɽɑ | ʋɑɑʔ"
    assert by_key[("27", "MOH")]["Manual_Transcription"] == "mut̪uɭ"
    assert by_key[("28", "LAD")]["Column"] == "left"
    assert by_key[("28", "MAD")]["Column"] == "right"
    assert by_key[("28", "SDI")]["Manual_Transcription"] == "silpiŋ | kɑpɑt"
    assert by_key[("29", "MOH")]["Manual_Transcription"] == "dʒulsɐhɑn"


def test_seventh_chunk_blanks_page_splits_and_dense_cells():
    rows = guard.load_manual_cells(LEDGER_030_034)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("30", "BAI")]["PDF_Page"] == "39"
    assert by_key[("30", "ORI")]["PDF_Page"] == "40"
    assert by_key[("30", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("31", "UDA")]["Manual_Transcription"] == "sʌsɑŋ | sil | sɑsɑŋɾeʔd dhiɾi"
    assert by_key[("31", "UDA")]["Source_Cognate_Labels"] == "1 | 3 | 1"
    assert by_key[("31", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("32", "SDI")]["Manual_Transcription"] == "tok | dɦusɾɑ"
    assert by_key[("33", "DUM")]["Column"] == "left"
    assert by_key[("33", "LAD")]["Column"] == "right"
    assert by_key[("33", "MDI")]["Manual_Transcription"] == "kuʈɑsi | hɑtɑoɽi"
    assert by_key[("34", "POD")]["Manual_Transcription"] == "tʃhuɾi | puŋki"


def test_eighth_chunk_qualifiers_page_splits_and_multiform_cells():
    rows = guard.load_manual_cells(LEDGER_035_039)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("35", "BAI")]["PDF_Page"] == "40"
    assert by_key[("35", "MJH")]["PDF_Page"] == "41"
    assert by_key[("35", "MCH")]["Manual_Transcription"] == "hɐke | hɔɽɑmhɑke"
    assert by_key[("35", "MCH")]["Uncertainty"].endswith("small | big")
    assert by_key[("35", "DIG")]["Manual_Transcription"] == "hɐke"
    assert "small" in by_key[("35", "DIG")]["Uncertainty"]
    assert by_key[("36", "SDI")]["Manual_Transcription"] == "bɑhɑɾi | bɑhɑɾi | boɽ"
    assert by_key[("38", "CHA")]["Column"] == "left"
    assert by_key[("38", "DIG")]["Column"] == "right"
    assert by_key[("39", "MDI")]["Manual_Transcription"] == "kitʃɾi | lidʒɑ | lugɑ"
    assert by_key[("39", "SDI")]["Manual_Transcription"] == "kitʃɾitʃ | lugɾi"


def test_ninth_chunk_blanks_page_splits_and_difficult_sequences():
    rows = guard.load_manual_cells(LEDGER_040_044)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("40", "BAI")]["PDF_Page"] == "41"
    assert by_key[("40", "MDH")]["PDF_Page"] == "42"
    assert by_key[("40", "POD")]["Manual_Transcription"] == "muɳd̪em"
    assert by_key[("41", "HDI")]["Review_Status"] == "source_blank"
    assert by_key[("41", "SDI")]["Manual_Transcription"] == "sin tʃɑndo | belɑ"
    assert by_key[("41", "SDI")]["Source_Cognate_Labels"] == "2 | 4"
    assert by_key[("42", "MCH")]["Manual_Transcription"] == "tʃɑnt̪uuʔ"
    assert by_key[("42", "HDI")]["Review_Status"] == "source_blank"
    assert by_key[("43", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("43", "DIG")]["Column"] == "left"
    assert by_key[("43", "DUM")]["Column"] == "right"
    assert by_key[("44", "ORI")]["Manual_Transcription"] == "t̪ɑɾɑ"


def test_tenth_chunk_page_break_variants_blanks_and_dental_marks():
    rows = guard.load_manual_cells(LEDGER_045_049)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("45", "UDA")]["Manual_Transcription"].split(" | ") == [
        "d̪ɑʔɑʔ gɑmɑ", "d̪ɑʔɑʔ", "d̪ɑʔɑʔ gɑmɑ",
    ]
    assert by_key[("45", "MDH")]["PDF_Page"] == "42-43"
    assert by_key[("45", "MDH")]["Source_Cognate_Labels"] == "1 | 2 | 2"
    assert by_key[("45", "SDI")]["Review_Status"] == "source_blank"
    assert by_key[("46", "HDI")]["Manual_Transcription"] == "d̪ɑɑʔ"
    assert by_key[("47", "POD")]["Manual_Transcription"] == "gʌdɑ"
    assert by_key[("48", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("48", "DUM")]["Column"] == "left"
    assert by_key[("48", "LAD")]["Column"] == "right"
    assert by_key[("48", "SDI")]["Manual_Transcription"] == "ɾimil | lɑhɾɑ"
    assert by_key[("49", "POD")]["Manual_Transcription"] == "itʃiɾ t̪ɑdɑ | bidʒlo"
    assert by_key[("49", "MDI")]["Manual_Transcription"] == "hitʃiɾ | t̪heɾ"
    assert by_key[("49", "SDI")]["Review_Status"] == "source_blank"


def test_eleventh_chunk_page_break_multiforms_and_repeated_dentals():
    rows = guard.load_manual_cells(LEDGER_050_054)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("50", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("50", "MDH")]["PDF_Page"] == "43"
    assert by_key[("50", "MJH")]["PDF_Page"] == "44"
    assert by_key[("50", "MCH")]["Manual_Transcription"] == "bɑnd̪ɑsike | lɔhɔɾbiŋ"
    assert by_key[("50", "ORI")]["Manual_Transcription"] == "ind̪ɾod̪ɑnɑsə"
    assert by_key[("51", "ORI")]["Manual_Transcription"] == "dʒhoɾɑkɑ"
    assert by_key[("52", "ORI")]["Manual_Transcription"] == "pət̪həɾə"
    assert by_key[("53", "MAD")]["Column"] == "left"
    assert by_key[("53", "MOH")]["Column"] == "right"
    assert by_key[("53", "SDI")]["Manual_Transcription"] == "hoɾ | sesɑ"
    assert by_key[("53", "ORI")]["Source_Cognate_Labels"] == "3 | 4"
    assert by_key[("54", "BAI")]["Manual_Transcription"] == "git̪il"
    assert by_key[("54", "ORI")]["Manual_Transcription"] == "bɑɭi"


def test_twelfth_chunk_nasalization_multiforms_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_055_059)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("55", "SDI")]["PDF_Page"] == "44"
    assert by_key[("55", "SNA")]["PDF_Page"] == "45"
    assert by_key[("55", "ORI")]["Manual_Transcription"] == "nĩːɑ"
    assert by_key[("56", "LAD")]["Manual_Transcription"] == "sukuɾ"
    assert by_key[("56", "MDI")]["Manual_Transcription"] == "sukul | dɦuŋgiɑ"
    assert by_key[("56", "HDI")]["Manual_Transcription"] == "mɔ̃ʔoʔ"
    assert by_key[("56", "SDI")]["Manual_Transcription"] == "dɦũɑ̃ | dɦuŋgiɑ"
    assert by_key[("57", "DUM")]["Manual_Transcription"] == "t̪oɾʌʔt̪"
    assert by_key[("58", "MOH")]["Column"] == "left"
    assert by_key[("58", "MUN")]["Column"] == "right"
    assert by_key[("58", "POD")]["Manual_Transcription"] == "losod | kɑd̪om"
    assert by_key[("59", "MCH")]["Manual_Transcription"] == "d̪ɑud̪ɑ"
    assert by_key[("59", "MJH")]["Manual_Transcription"] == "gund̪ɑ"


def test_thirteenth_chunk_root_variants_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_060_064)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("60", "DIG")]["Manual_Transcription"] == "sɛmɔŋɔm"
    assert by_key[("60", "SDI")]["Manual_Transcription"] == "sonɑ | sɑmɑɾom"
    assert by_key[("60", "ORI")]["Manual_Transcription"] == "sunːɑ"
    assert by_key[("61", "DUM")]["Manual_Transcription"] == "d̪ʌɾu"
    assert by_key[("61", "ORI")]["Manual_Transcription"] == "gɑtʃhɑ"
    assert by_key[("62", "HDI")]["Manual_Transcription"] == "sɛkɛm"
    assert by_key[("63", "UDA")]["Column"] == "left"
    assert by_key[("63", "MCH")]["Column"] == "right"
    assert by_key[("63", "MJH")]["Manual_Transcription"] == "ɾeʔheʔ | ɾeʔɾ"
    assert by_key[("63", "HDI")]["Manual_Transcription"] == "tʃeɾoɾeʔ"
    assert by_key[("64", "ORI")]["Manual_Transcription"] == "kont̪ɑ"


def test_fourteenth_chunk_retroflex_laterals_blanks_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_065_069)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("65", "LAD")]["Manual_Transcription"] == "bɑː"
    assert by_key[("66", "BAI")]["PDF_Page"] == "46"
    assert by_key[("66", "CHA")]["PDF_Page"] == "47"
    assert by_key[("67", "UDA")]["Manual_Transcription"] == "uɭi"
    assert by_key[("68", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("68", "MJH")]["Column"] == "left"
    assert by_key[("68", "HDI")]["Column"] == "right"
    assert by_key[("68", "HDI")]["Manual_Transcription"] == "ked̪eɭ"
    assert by_key[("68", "SDI")]["Review_Status"] == "source_blank"
    assert by_key[("69", "DIG")]["Manual_Transcription"] == "gɔhɔmo"
    assert by_key[("69", "ORI")]["Manual_Transcription"] == "gohomõ"


def test_fifteenth_chunk_retroflexes_blanks_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_070_074)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("70", "DIG")]["Manual_Transcription"] == "t̪eɾbudʒ"
    assert by_key[("70", "SNA")]["Review_Status"] == "source_blank"
    assert by_key[("71", "MAD")]["PDF_Page"] == "47"
    assert by_key[("71", "MOH")]["PDF_Page"] == "48"
    assert by_key[("71", "UDA")]["Manual_Transcription"] == "mɑɳɖi"
    assert by_key[("72", "DUM")]["Manual_Transcription"] == "golɑɭui"
    assert by_key[("73", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("73", "ORI")]["Manual_Transcription"] == "bɑiŋgoɾõ"
    assert by_key[("74", "MJH")]["Manual_Transcription"] == "muɸuli"
    assert by_key[("74", "HDI")]["Manual_Transcription"] == "bɛɖɛm"


def test_sixteenth_chunk_rhotics_nasals_blanks_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_075_079)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("75", "ORI")]["Manual_Transcription"] == "məɾitʃə"
    assert by_key[("76", "MCH")]["PDF_Page"] == "48"
    assert by_key[("76", "MDI")]["PDF_Page"] == "49"
    assert by_key[("76", "MUN")]["Manual_Transcription"] == "sɛsɑn"
    assert by_key[("77", "DUM")]["Manual_Transcription"] == "ɾʌsuɲĩ"
    assert by_key[("77", "HDI")]["Manual_Transcription"] == "ɾɛsuiŋ"
    assert by_key[("78", "LAD")]["Manual_Transcription"] == "pjɑdʒi"
    assert by_key[("79", "DIG")]["Column"] == "left"
    assert by_key[("79", "DUM")]["Column"] == "right"
    assert by_key[("79", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("79", "ORI")]["Manual_Transcription"] == "phul kobi"


def test_seventeenth_chunk_continuations_blanks_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_080_084)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("80", "MJH")]["Manual_Transcription"] == "bilɛʈi | pɛʈɛl"
    assert by_key[("80", "SDI")]["Review_Status"] == "source_blank"
    assert by_key[("81", "HDI")]["PDF_Page"] == "49"
    assert by_key[("81", "SDI")]["PDF_Page"] == "50"
    assert by_key[("81", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("82", "ORI")]["Manual_Transcription"] == "t̪elo"
    assert by_key[("83", "ORI")]["Manual_Transcription"] == "luɳə | nũno"
    assert by_key[("84", "MAD")]["Column"] == "left"
    assert by_key[("84", "MOH")]["Column"] == "right"
    assert by_key[("84", "SDI")]["Manual_Transcription"] == "beɾel dʒel | dʒel"


def test_eighteenth_chunk_retroflexes_continuations_and_column_split():
    rows = guard.load_manual_cells(LEDGER_085_089)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("85", "BAI")]["Manual_Transcription"] == "iʈil"
    assert by_key[("86", "ORI")]["Manual_Transcription"] == "mɑːtʃo"
    assert by_key[("87", "ORI")]["Manual_Transcription"] == "kukudɑ"
    assert by_key[("88", "LAD")]["Manual_Transcription"] == "ʌɳɖʌ"
    assert by_key[("88", "MDI")]["Manual_Transcription"] == "dʒɑɾom | bili"
    assert by_key[("89", "POD")]["Column"] == "left"
    assert by_key[("89", "UDA")]["Column"] == "right"
    assert by_key[("89", "POD")]["Manual_Transcription"] == "gei | uɾi"
    assert by_key[("89", "MDI")]["Manual_Transcription"] == "gai | gundi"
    assert by_key[("89", "SDI")]["Manual_Transcription"] == "gai | dɑŋgri"


def test_nineteenth_chunk_nasalization_continuations_and_page_splits():
    rows = guard.load_manual_cells(LEDGER_090_094)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("90", "CHA")]["Manual_Transcription"] == "kiɖɑ | mɔ̃s"
    assert by_key[("90", "POD")]["Manual_Transcription"] == "keɖɑ | mɔ̃isi"
    assert by_key[("90", "MDI")]["Manual_Transcription"] == "keɖɑ | birkeɾɑ"
    assert by_key[("91", "SDI")]["PDF_Page"] == "51"
    assert by_key[("91", "SNA")]["PDF_Page"] == "52"
    assert by_key[("92", "SDI")]["Manual_Transcription"] == "siŋgɑ | ɖɑbe"
    assert by_key[("93", "SDI")]["Manual_Transcription"] == "tʃɑnɖbol"
    assert by_key[("94", "MOH")]["Column"] == "left"
    assert by_key[("94", "MUN")]["Column"] == "right"
    assert by_key[("94", "ORI")]["Manual_Transcription"] == "tʃheɭi"


def test_twentieth_chunk_glottals_blanks_continuations_and_column_split():
    rows = guard.load_manual_cells(LEDGER_095_099)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("95", "DIG")]["Manual_Transcription"] == "sɛʈɑʔ"
    assert by_key[("96", "SDI")]["Manual_Transcription"] == "biŋ | kɑl"
    assert by_key[("97", "LAD")]["Manual_Transcription"] == "gɑɖi | hanumɑn"
    assert by_key[("97", "MUN")]["Review_Status"] == "source_blank"
    assert by_key[("97", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("98", "SDI")]["Manual_Transcription"] == "sikɾĩtʃ"
    assert by_key[("99", "UDA")]["Column"] == "left"
    assert by_key[("99", "MCH")]["Column"] == "right"
    assert by_key[("99", "BAI")]["Manual_Transcription"] == "mũʔi"
    assert by_key[("99", "SDI")]["Manual_Transcription"] == "mutʃʔ"


def test_twenty_first_chunk_retroflex_nasals_continuations_and_column_split():
    rows = guard.load_manual_cells(LEDGER_100_104)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("100", "CHA")]["Manual_Transcription"] == "ʈʌɳʈulɑ"
    assert by_key[("100", "POD")]["Manual_Transcription"] == "ʈɑɳʈɑle"
    assert by_key[("100", "SNA")]["Manual_Transcription"] == "binʈɪ"
    assert by_key[("100", "ORI")]["Manual_Transcription"] == "buɖɦiɑɳi"
    assert by_key[("101", "MDI")]["Manual_Transcription"] == "nuʈum | num"
    assert by_key[("102", "MDI")]["Manual_Transcription"] == "hoɾo | koɾɑ"
    assert by_key[("103", "ORI")]["Manual_Transcription"] == "st̪ɾi"
    assert by_key[("104", "UDA")]["Column"] == "left"
    assert by_key[("104", "MCH")]["Column"] == "right"
    assert by_key[("104", "SDI")]["Manual_Transcription"] == "giɖɾɑ"
    assert by_key[("104", "SNA")]["Manual_Transcription"] == "giɖɾə"


def test_twenty_second_chunk_kin_terms_nasal_places_and_continuations():
    rows = guard.load_manual_cells(LEDGER_105_109)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("105", "MDI")]["Manual_Transcription"] == "ɑbɑ | ɑpu"
    assert by_key[("105", "SDI")]["Manual_Transcription"] == "ɑpɑ | bɑ"
    assert by_key[("106", "CHA")]["Manual_Transcription"] == "mɑɳ"
    assert by_key[("106", "SDI")]["Manual_Transcription"] == "eŋgɑ | ɑyo"
    assert by_key[("107", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("107", "BAI")]["Manual_Transcription"] == "mɑɾɑɳɖɑɖɑ"
    assert by_key[("108", "MAD")]["Manual_Transcription"] == "huɖiɲɳi"
    assert by_key[("108", "UDA")]["Manual_Transcription"] == "huɖiɳɖɑɖɑ"
    assert by_key[("109", "MCH")]["Column"] == "left"
    assert by_key[("109", "MDI")]["Column"] == "right"
    assert by_key[("109", "SDI")]["Manual_Transcription"] == "ɖɑi | ɑdʒi"
    assert by_key[("109", "ORI")]["Manual_Transcription"] == "nɑnːi | ɖiɖi"


def test_twenty_third_chunk_kin_terms_blanks_and_cross_column_continuations():
    rows = guard.load_manual_cells(LEDGER_110_114)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("110", "LAD")]["Manual_Transcription"] == "uɾiŋmisi"
    assert by_key[("110", "SNA")]["Manual_Transcription"] == "hepɔn mɑi"
    assert by_key[("111", "SDI")]["Manual_Transcription"] == "hon | koɾɑ hopon"
    assert by_key[("111", "ORI")]["Manual_Transcription"] == "puːo"
    assert by_key[("112", "MAD")]["Manual_Transcription"] == "kuɖihoniɾɑ"
    assert by_key[("113", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("113", "HDI")]["Review_Status"] == "source_blank"
    assert by_key[("113", "SDI")]["Manual_Transcription"] == "dʒɑ̃wɑ̃e | heɾel"
    assert by_key[("114", "POD")]["Column"] == "left"
    assert by_key[("114", "UDA")]["Column"] == "right"
    assert by_key[("114", "UDA")]["Manual_Transcription"] == "iɾɑ | buɖi"
    assert by_key[("114", "MDI")]["Manual_Transcription"] == "kuɾi | oɾɑ hoɾo"
    assert by_key[("114", "MJH")]["Review_Status"] == "source_blank"
    assert by_key[("114", "HDI")]["Review_Status"] == "source_blank"


def test_twenty_fourth_chunk_age_and_time_terms_with_continuations():
    rows = guard.load_manual_cells(LEDGER_115_119)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("115", "DIG")]["Manual_Transcription"] == "huɖiɲhõn"
    assert by_key[("115", "MCH")]["Manual_Transcription"] == "dɦɑŋgeɾɑ"
    assert by_key[("116", "ORI")]["Manual_Transcription"] == "dʒio pilɑʔ"
    assert by_key[("117", "MDI")]["Manual_Transcription"] == "ɖin | hulɑŋ"
    assert by_key[("118", "DUM")]["Manual_Transcription"] == "nĩɖe"
    assert by_key[("118", "SDI")]["Manual_Transcription"] == "nindɑ | nindɑ"
    assert by_key[("118", "ORI")]["Manual_Transcription"] == "ɾɑt̪i"
    assert by_key[("119", "POD")]["Column"] == "left"
    assert by_key[("119", "UDA")]["Column"] == "right"
    assert by_key[("119", "LAD")]["Manual_Transcription"] == "siʈːɑ"
    assert by_key[("119", "MDI")]["Manual_Transcription"] == "setɑ | idɑŋ"
    assert by_key[("119", "ORI")]["Manual_Transcription"] == "səkɑɭə"


def test_twenty_fifth_chunk_time_terms_dentals_and_repeated_responses():
    rows = guard.load_manual_cells(LEDGER_120_124)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("120", "CHA")]["Manual_Transcription"] == "dɦupʌɾ"
    assert by_key[("120", "ORI")]["Manual_Transcription"] == "məd̪ɦjɑnə"
    assert by_key[("121", "POD")]["Manual_Transcription"] == "ʌub siŋgi | ʌub siŋgi"
    assert by_key[("121", "ORI")]["Manual_Transcription"] == "sənd̪ɦjɑ"
    assert by_key[("122", "UDA")]["Manual_Transcription"] == "holo"
    assert by_key[("123", "BAI")]["Manual_Transcription"] == "t̪isiŋ | t̪isiŋ"
    assert by_key[("123", "BAI")]["Source_Cognate_Labels"] == "1 | 2"
    assert by_key[("123", "MDH")]["Manual_Transcription"] == "t̪isiŋ | t̪isiŋ"
    assert by_key[("123", "MJH")]["Manual_Transcription"] == "isiŋ"
    assert by_key[("123", "SNA")]["Manual_Transcription"] == "t̪ɛheŋ"
    assert by_key[("124", "ORI")]["Manual_Transcription"] == "ɑsont̪ɑ kɑli"


def test_twenty_sixth_chunk_calendar_terms_contrasts_and_nasalization():
    rows = guard.load_manual_cells(LEDGER_125_129)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("125", "DIG")]["Manual_Transcription"] == "ɛt̪əuɑɾi"
    assert by_key[("125", "UDA")]["Manual_Transcription"] == "sɛpt̪ɑ | hɑt"
    assert by_key[("125", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("126", "LAD")]["Manual_Transcription"] == "tʃɑnd̪uʔ"
    assert by_key[("126", "MCH")]["Manual_Transcription"] == "mid̪tʃɑnɖuʔu"
    assert by_key[("126", "MDI")]["Manual_Transcription"] == "tʃɑnɖu"
    assert by_key[("126", "MCH")]["PDF_Page"] == "59"
    assert by_key[("127", "SDI")]["Manual_Transcription"] == "seɾmɑ | botʃhoɾ"
    assert by_key[("128", "LAD")]["Manual_Transcription"] == "puɾnːɑ"
    assert by_key[("129", "DIG")]["Manual_Transcription"] == "nɑ̃uɑ̃"
    assert by_key[("129", "MDH")]["Manual_Transcription"] == "nɑmɑ | nɑwɑ"
    assert by_key[("129", "ORI")]["Manual_Transcription"] == "nuɑ̃"


def test_twenty_seventh_chunk_adjectives_retroflexes_and_cross_column_cell():
    rows = guard.load_manual_cells(LEDGER_130_134)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("130", "UDA")]["Manual_Transcription"] == "bes | bigi"
    assert by_key[("130", "MCH")]["Manual_Transcription"] == "bes | bugin"
    assert by_key[("131", "DIG")]["Manual_Transcription"] == "dʒuɖɑ"
    assert by_key[("131", "MCH")]["Manual_Transcription"] == "eʔʈkɑ"
    assert by_key[("131", "MOH")]["PDF_Page"] == "60"
    assert by_key[("132", "UDA")]["Manual_Transcription"] == "oɖɑɖ | lejeɾ"
    assert by_key[("133", "DIG")]["Manual_Transcription"] == "ɾoɭo"
    assert by_key[("133", "SDI")]["Manual_Transcription"] == "hindʒit | tʃuttʃɑt"
    assert by_key[("133", "SDI")]["Column"] == "left/right"
    assert by_key[("134", "BAI")]["Manual_Transcription"] == "dʒiliŋ"
    assert by_key[("134", "SDI")]["Manual_Transcription"] == "dʒelen | dʒɦɑɭ"


def test_twenty_eighth_chunk_temperature_direction_and_dental_contrasts():
    rows = guard.load_manual_cells(LEDGER_135_139)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("135", "UDA")]["Manual_Transcription"] == "khɑ̃diɑ"
    assert by_key[("135", "MDI")]["Manual_Transcription"] == "huɾiŋ | t̪um"
    assert by_key[("135", "SDI")]["Manual_Transcription"] == "khɑto | geɖɑ"
    assert by_key[("136", "MDI")]["Manual_Transcription"] == "lolo | dʒete"
    assert by_key[("136", "LAD")]["PDF_Page"] == "61"
    assert by_key[("137", "MCH")]["Manual_Transcription"] == "t̪ut̪ukun"
    assert by_key[("137", "POD")]["Manual_Transcription"] == "ɾijʌd | ɾʌbʌn"
    assert by_key[("138", "CHA")]["Manual_Transcription"] == "mɑndi kuʈi"
    assert by_key[("138", "SDI")]["Column"] == "right"
    assert by_key[("139", "HDI")]["Manual_Transcription"] == "liŋpt̪i"
    assert by_key[("139", "ORI")]["Manual_Transcription"] == "bɑːmo"


def test_twenty_ninth_chunk_distance_size_weight_and_nasal_contrasts():
    rows = guard.load_manual_cells(LEDGER_140_144)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("140", "MDI")]["Manual_Transcription"] == "nipɑt̪"
    assert by_key[("140", "ORI")]["Manual_Transcription"] == "pɑkːo"
    assert by_key[("141", "HDI")]["Manual_Transcription"] == "sɑɳiŋ"
    assert by_key[("141", "MUN")]["PDF_Page"] == "62"
    assert by_key[("142", "ORI")]["Manual_Transcription"] == "boɽo"
    assert by_key[("143", "SDI")]["Manual_Transcription"] == "huɖiŋ | kɑtitʃʔ"
    assert by_key[("143", "ORI")]["Manual_Transcription"] == "sɑnõ"
    assert by_key[("144", "MOH")]["Manual_Transcription"] == "t̪egɑɖɑ"
    assert by_key[("144", "ORI")]["Manual_Transcription"] == "bɦɑɾi"


def test_thirtieth_chunk_continuations_dentals_retroflexes_and_blank():
    rows = guard.load_manual_cells(LEDGER_145_149)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("145", "POD")]["Manual_Transcription"] == "lʌbɑɾ | ɾʌbɑl"
    assert by_key[("145", "SDI")]["Manual_Transcription"] == "ɾɑwɑl | mɑɾsɑl"
    assert by_key[("145", "SNA")]["Manual_Transcription"] == "ɾeʋɑl"
    assert by_key[("145", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("146", "POD")]["PDF_Page"] == "62"
    assert by_key[("146", "UDA")]["PDF_Page"] == "63"
    assert by_key[("146", "SDI")]["Manual_Transcription"] == "tʃet̪ɑn | tʃot"
    assert by_key[("147", "ORI")]["Manual_Transcription"] == "t̪ələ"
    assert by_key[("148", "DIG")]["Manual_Transcription"] == "phuɳɖi"
    assert by_key[("148", "ORI")]["Manual_Transcription"] == "ɖholɑ"
    assert by_key[("149", "LAD")]["Manual_Transcription"] == "heɲd̪ɛ"
    assert by_key[("149", "ORI")]["Manual_Transcription"] == "koɭɑʔ"


def test_thirty_first_chunk_numerals_continuations_and_page_split():
    rows = guard.load_manual_cells(LEDGER_150_154)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("150", "ORI")]["Manual_Transcription"] == "ɾoŋgo | nɑli"
    assert by_key[("151", "UDA")]["PDF_Page"] == "63"
    assert by_key[("151", "MCH")]["PDF_Page"] == "64"
    assert by_key[("151", "CHA")]["Manual_Transcription"] == "mõe"
    assert by_key[("151", "LAD")]["Manual_Transcription"] == "mijʌnd̪ʔ"
    assert by_key[("151", "SNA")]["Manual_Transcription"] == "mit̪ɑŋ"
    assert by_key[("152", "MDI")]["Manual_Transcription"] == "bɑɾ | bɑɾiɑ"
    assert by_key[("153", "MDH")]["Manual_Transcription"] == "t̪in"
    assert by_key[("154", "MCH")]["Manual_Transcription"] == "opuɲie"
    assert by_key[("154", "ORI")]["Manual_Transcription"] == "tʃɑɾi"


def test_thirty_second_chunk_numerals_nasalization_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_155_159)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("155", "DUM")]["Manual_Transcription"] == "mõnẽɑ"
    assert by_key[("155", "SDI")]["Manual_Transcription"] == "mõɾẽ"
    assert by_key[("156", "MCH")]["PDF_Page"] == "64"
    assert by_key[("156", "MDI")]["PDF_Page"] == "65"
    assert by_key[("156", "MDI")]["Manual_Transcription"] == "t̪uɾiɑ | t̪uɾuiɑ"
    assert by_key[("157", "MCH")]["Manual_Transcription"] == "ejeː"
    assert by_key[("158", "MDI")]["Manual_Transcription"] == "iɾɑliɑ | iɾiliɑ"
    assert by_key[("158", "MJH")]["Review_Status"] == "source_blank"
    assert by_key[("158", "SDI")]["Manual_Transcription"] == "iɾɑɭ"
    assert by_key[("159", "DIG")]["Manual_Transcription"] == "nõ"


def test_thirty_third_chunk_compound_numerals_continuations_and_blanks():
    rows = guard.load_manual_cells(LEDGER_160_164)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("160", "MDI")]["Manual_Transcription"] == "gel | geleɑ"
    assert by_key[("160", "MJH")]["Review_Status"] == "source_blank"
    assert by_key[("161", "UDA")]["PDF_Page"] == "65"
    assert by_key[("161", "MCH")]["PDF_Page"] == "66"
    assert by_key[("161", "LAD")]["Manual_Transcription"] == "gel mijʌd̪ʔ"
    assert by_key[("162", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("163", "MOH")]["Manual_Transcription"] == "kudije | bis"
    assert by_key[("163", "UDA")]["Manual_Transcription"] == "mot hisi | kodi"
    assert by_key[("163", "MCH")]["Manual_Transcription"] == "mid̪isi"
    assert by_key[("164", "MCH")]["Manual_Transcription"] == "mod̪ehisi"


def test_thirty_fourth_chunk_questions_dentals_continuations_and_blank():
    rows = guard.load_manual_cells(LEDGER_165_169)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("165", "DIG")]["Manual_Transcription"] == "ɔkɑje"
    assert by_key[("166", "MCH")]["PDF_Page"] == "66"
    assert by_key[("166", "MDI")]["PDF_Page"] == "67"
    assert by_key[("166", "LAD")]["Manual_Transcription"] == "kɑɲɑ"
    assert by_key[("167", "ORI")]["Manual_Transcription"] == "keuntɑɾe | kuɑde"
    assert by_key[("168", "DUM")]["Review_Status"] == "source_blank"
    assert by_key[("168", "SDI")]["Manual_Transcription"] == "tisɾe | khɑn"
    assert by_key[("168", "SNA")]["Manual_Transcription"] == "t̪iso"
    assert by_key[("169", "SNA")]["Manual_Transcription"] == "t̪inɛŋ"
    assert by_key[("169", "ORI")]["Manual_Transcription"] == "ket̪e"


def test_thirty_fifth_chunk_demonstratives_continuations_and_page_split():
    rows = guard.load_manual_cells(LEDGER_170_174)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("170", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("170", "POD")]["Manual_Transcription"] == "tʃilikɑnɑ | tʃiminprɑkɑɾ"
    assert by_key[("170", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("170", "ORI")]["Manual_Transcription"] == "kemit̪i"
    assert by_key[("171", "UDA")]["PDF_Page"] == "67"
    assert by_key[("171", "MCH")]["PDF_Page"] == "68"
    assert by_key[("171", "SDI")]["Manual_Transcription"] == "niɑ | noɑ"
    assert by_key[("172", "CHA")]["Manual_Transcription"] == "hɑnɑ | inɑ"
    assert by_key[("172", "DUM")]["Manual_Transcription"] == "hʌːe"
    assert by_key[("173", "MDI")]["Manual_Transcription"] == "neɑko | niku"
    assert by_key[("173", "SDI")]["Manual_Transcription"] == "noɑko | noko"
    assert by_key[("174", "MUN")]["Manual_Transcription"] == "hɑnt̪ɑi"
    assert by_key[("174", "ORI")]["Manual_Transcription"] == "seisɑbu"


def test_thirty_sixth_chunk_quantifiers_continuations_and_dental_marks():
    rows = guard.load_manual_cells(LEDGER_175_179)
    assert len(rows) == 5 * 18 == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("175", "DIG")]["Manual_Transcription"] == "sɛmɑn | sɔmɑn"
    assert by_key[("175", "POD")]["Manual_Transcription"] == "sʌmɑn | motgiɑ"
    assert by_key[("175", "MJH")]["Manual_Transcription"] == "mit̪gi"
    assert by_key[("176", "MOH")]["PDF_Page"] == "68"
    assert by_key[("176", "MUN")]["PDF_Page"] == "69"
    assert by_key[("176", "POD")]["Manual_Transcription"] == "bɦenɑ bɦenɑ | begɑɾ begɑɾ"
    assert by_key[("176", "MDI")]["Manual_Transcription"] == "et̪ɑ | kilimili"
    assert by_key[("177", "DUM")]["Manual_Transcription"] == "best̪iɑ"
    assert by_key[("177", "MCH")]["Manual_Transcription"] == "gɔt̪ɑ | soben"
    assert by_key[("178", "LAD")]["Manual_Transcription"] == "ɾɑpɑ̃tʔn̩"
    assert by_key[("178", "MCH")]["Manual_Transcription"] == "ɾɑpud̪"
    assert by_key[("178", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("178", "SDI")]["Manual_Transcription"] == "bɦɑŋgɑ | kɑtʃɑ | t̪ut̪ɑ"
    assert by_key[("178", "SNA")]["Column"] == "right"
    assert by_key[("179", "SDI")]["Manual_Transcription"] == "thoɾɑ gɑn | ekɑ | dukɑ"


def test_thirty_seventh_chunk_quantifiers_continuations_and_page_break():
    rows = guard.load_manual_cells(LEDGER_180_181)
    assert len(rows) == 2 * 18 == 36
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("180", "DUM")]["Manual_Transcription"] == "bedʒʌŋ | d̪eheɾ"
    assert by_key[("180", "POD")]["Manual_Transcription"] == "bidʒen | puɾe"
    assert by_key[("180", "MDI")]["Manual_Transcription"] == "d̪ɦeɾ | ɑn hut | isu"
    assert by_key[("180", "SDI")]["Manual_Transcription"] == "ɑemɑ | ɑdi"
    assert by_key[("180", "BAI")]["PDF_Page"] == "69"
    assert by_key[("181", "BAI")]["PDF_Page"] == "70"
    assert by_key[("181", "BAI")]["Manual_Transcription"] == "dʒʌt̪ɔ"
    assert by_key[("181", "MAD")]["Manual_Transcription"] == "dʒʌnt̪o"
    assert by_key[("181", "MJH")]["Manual_Transcription"] == "t̪himbɑgi"
    assert by_key[("181", "SDI")]["Manual_Transcription"] == "dʒot̪o | sɑnɑm"


def test_thirty_eighth_chunk_eat_pair_continuation_and_blank():
    rows = guard.load_manual_cells(LEDGER_182)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("182", "CHA")]["Manual_Transcription"] == (
        "dʒomeme, dʒomkijɑ | dʒʌmem, dzʌmkeɑj"
    )
    assert by_key[("182", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("182", "LAD")]["Manual_Transcription"] == "dʒomʌm, dʒomled̪ɑ"
    assert by_key[("182", "MUN")]["Manual_Transcription"] == "dʒomem, nukud̪e"
    assert by_key[("182", "HDI")]["Manual_Transcription"] == "dʒomem, dʒɔŋkid̪ɑ"
    assert by_key[("182", "ORI")]["Manual_Transcription"] == "kɑibə"


def test_thirty_ninth_chunk_bite_pair_column_split_and_blank():
    rows = guard.load_manual_cells(LEDGER_183)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("183", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("183", "DUM")]["Manual_Transcription"] == "huegidʒiʔme, hueligit̪"
    assert by_key[("183", "LAD")]["Manual_Transcription"] == "huwɑgiʔme, huwɑkiʌ"
    assert by_key[("183", "POD")]["Manual_Transcription"] == "hueegʔme, hueʔkidʒijɑʔ"
    assert by_key[("183", "UDA")]["Column"] == "right"
    assert by_key[("183", "HDI")]["Manual_Transcription"] == "huʔjɑjɛm, hujekid̪ɑ"
    assert by_key[("183", "SDI")]["Manual_Transcription"] == "geɾ | lɑsok"
    assert by_key[("183", "SDI")]["Source_Cognate_Labels"] == "2 | 5"


def test_fortieth_chunk_hunger_predicates_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_184)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("184", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("184", "LAD")]["Manual_Transcription"] == "ɾɛŋgɛʔi t̪aikiŋ"
    assert by_key[("184", "MOH")]["Manual_Transcription"] == (
        "ɾɛŋkedʒekanɑ, ɾɛŋkedʒɑkɑnɑ"
    )
    assert by_key[("184", "MCH")]["Manual_Transcription"] == (
        "ɾeŋge dʒɑʔɑjə, ɾeŋgeʔelijə"
    )
    assert by_key[("184", "MJH")]["Manual_Transcription"] == (
        "ɾɛŋkekgi mɛnɛdʒɑ, ɾɛŋkekgi t̪ɑin kɑnɑi"
    )
    assert by_key[("184", "SNA")]["Manual_Transcription"] == (
        "ɾɛŋke mɛnɑjɑ, ɾɛŋket̪et̪ɑhinkinɑj"
    )
    assert by_key[("184", "ORI")]["Manual_Transcription"] == "bɦoko helɑʔ"


def test_forty_first_chunk_drink_pair_page_break_nasalization_and_blank():
    rows = guard.load_manual_cells(LEDGER_185)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("185", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("185", "DUM")]["Manual_Transcription"] == "nuiẽme, nulijɑ"
    assert by_key[("185", "LAD")]["Manual_Transcription"] == "nuʔĩme, nuled̪ɑ"
    assert by_key[("185", "MAD")]["Manual_Transcription"] == "nuitme, nukijɑt̪"
    assert by_key[("185", "POD")]["Manual_Transcription"] == "nuʔme, nuiliʌŋ"
    assert by_key[("185", "SDI")]["Manual_Transcription"] == "nũ"
    assert by_key[("185", "SDI")]["PDF_Page"] == "70"
    assert by_key[("185", "SNA")]["PDF_Page"] == "71"
    assert by_key[("185", "ORI")]["Manual_Transcription"] == "piːbɑ"


def test_forty_second_chunk_thirst_predicates_repeated_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_186)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("186", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("186", "CHA")]["Manual_Transcription"] == "t̪it̪ɑŋt̪ɑdʒi t̪aikenɑ"
    assert by_key[("186", "LAD")]["Manual_Transcription"] == "t̪ɛt̪ɑŋt̪aikinʌ"
    assert by_key[("186", "MOH")]["Manual_Transcription"] == (
        "t̪ɛt̪ɑŋt̪edʒijɑ, hɔlɑ t̪ɛt̪ɑŋt̪edʒijɑ"
    )
    assert by_key[("186", "MCH")]["Manual_Transcription"] == (
        "t̪ɛt̪ɑŋdʒɑʔɑje, t̪ɛt̪ɑŋliʔɑ"
    )
    assert by_key[("186", "MJH")]["Manual_Transcription"] == (
        "t̪ɛt̪ɑŋɔt̪ɛne, t̪ɛt̪ɑŋɔt̪ɛine"
    )
    assert by_key[("186", "SNA")]["Manual_Transcription"] == (
        "t̪ɛt̪ɑŋikɑnɑ, t̪ɛt̪ɑŋlid̪ijɑ"
    )


def test_forty_third_chunk_sleep_pair_column_break_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_187)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("187", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("187", "DUM")]["Manual_Transcription"] == "git̪iʔt̪me, git̪iʔt̪linɑ"
    assert by_key[("187", "LAD")]["Manual_Transcription"] == "duɾʌmmeʔ, duɾʌmlɛŋɑ"
    assert by_key[("187", "MOH")]["Manual_Transcription"] == (
        "d̪ud̪umke, edʒ d̪ud̪ud̪umlenɑ"
    )
    assert by_key[("187", "SDI")]["Manual_Transcription"] == "git̪itʃʔ | dʒɑpit"
    assert by_key[("187", "SDI")]["Source_Cognate_Labels"] == "1 | 3"
    assert by_key[("187", "SNA")]["Column"] == "right"
    assert by_key[("187", "ORI")]["Manual_Transcription"] == "nido"


def test_forty_fourth_chunk_lie_down_multiforms_dentals_and_two_blanks():
    rows = guard.load_manual_cells(LEDGER_188)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("188", "BAI")]["Review_Status"] == "source_blank"
    assert by_key[("188", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("188", "LAD")]["Manual_Transcription"] == (
        "git̪ime, git̪iɑkʌn t̪aikinʌ"
    )
    assert by_key[("188", "POD")]["Manual_Transcription"] == (
        "git̪itme, git̪itdʒɑnɑe | git̪iʔmeʔ, git̪id̪linɑ"
    )
    assert by_key[("188", "POD")]["Source_Cognate_Labels"] == "1 | 1"
    assert by_key[("188", "MDI")]["Manual_Transcription"] == "git̪i | bɑt̪in | buɾum"
    assert by_key[("188", "SNA")]["Manual_Transcription"] == (
        "gud̪t̪o hɛnt̪ɑd̪ope, gud̪t̪owenɑj"
    )
    assert by_key[("188", "ORI")]["Manual_Transcription"] == "poɾigolɑ"


def test_forty_fifth_chunk_sit_down_dentals_beta_and_blank():
    rows = guard.load_manual_cells(LEDGER_189)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("189", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("189", "LAD")]["Manual_Transcription"] == (
        "d̪ubʔmeʔ, d̪uβɑkɑn tɑikinʌ"
    )
    assert by_key[("189", "MAD")]["Manual_Transcription"] == (
        "d̪ud̪upme, d̪ud̪up dʒɑnɑe"
    )
    assert by_key[("189", "MOH")]["Manual_Transcription"] == (
        "d̪ud̪uʔme, edʒd̪d̪uʔlenɑ"
    )
    assert by_key[("189", "POD")]["Manual_Transcription"] == "dud̪upme, dud̪uplinɑ"
    assert by_key[("189", "SNA")]["Manual_Transcription"] == "d̪uluʔme, d̪uluʔjenɑi"
    assert by_key[("189", "ORI")]["Manual_Transcription"] == "bosibɑ"


def test_forty_sixth_chunk_give_multiforms_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_190)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("190", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("190", "LAD")]["Manual_Transcription"] == "umɑŋmeʔ, omlɛd̪ɑ"
    assert by_key[("190", "MOH")]["Manual_Transcription"] == (
        "dɑ imɑŋme, omɑ dʒiɑt | ɔmem, mɑdʒije"
    )
    assert by_key[("190", "UDA")]["Source_Cognate_Labels"] == "1 | 2"
    assert by_key[("190", "MDI")]["Manual_Transcription"] == "em | om"
    assert by_key[("190", "MDH")]["Manual_Transcription"] == (
        "d̪e emonme, omkid̪ɑ | d̪e emonme, omkid̪ɑ"
    )
    assert by_key[("190", "SNA")]["Manual_Transcription"] == "iŋimɛŋpe, ɛmɑd̪iŋɑj"
    assert by_key[("190", "ORI")]["Manual_Transcription"] == "d̪ebɑ"


def test_forty_seventh_chunk_burn_multiforms_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_191)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("191", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("191", "LAD")]["Manual_Transcription"] == "lot̪ʔn̩me, loleŋɑ"
    assert by_key[("191", "MUN")]["Manual_Transcription"] == (
        "dʒult̪enɑ, hɛt̪ɑrdʒɑnɑ"
    )
    assert by_key[("191", "MDI")]["Manual_Transcription"] == "lo | ɑt̪ɑɾ"
    assert by_key[("191", "HDI")]["Manual_Transcription"] == (
        "dʒult̪inɑ, dʒulkid̪ɑ"
    )
    assert by_key[("191", "SDI")]["Manual_Transcription"] == "lo | dʒeɾet"
    assert by_key[("191", "ORI")]["Manual_Transcription"] == "dʒolibɑ"


def test_forty_eighth_chunk_die_pair_column_break_glottals_and_blank():
    rows = guard.load_manual_cells(LEDGER_192)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("192", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("192", "CHA")]["Manual_Transcription"] == (
        "gudʒuʔme, godʒʔdʒɑnɑe"
    )
    assert by_key[("192", "BAI")]["Column"] == "left"
    assert by_key[("192", "DIG")]["Column"] == "right"
    assert by_key[("192", "MCH")]["Manual_Transcription"] == (
        "goʔedʒenɑ, goʔedʒɑnɑ"
    )
    assert by_key[("192", "SDI")]["Manual_Transcription"] == "gudʒuk, gotʃʔ"
    assert by_key[("192", "ORI")]["Manual_Transcription"] == "moɾonõ"


def test_forty_ninth_chunk_kill_pair_dentals_glottals_and_blank():
    rows = guard.load_manual_cells(LEDGER_193)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("193", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("193", "MAD")]["Manual_Transcription"] == "goet̪kijɑe"
    assert by_key[("193", "MOH")]["Manual_Transcription"] == "gudʒije, godʒt̪edʒije"
    assert by_key[("193", "MCH")]["Manual_Transcription"] == "d̪ɑlie, d̪elkie"
    assert by_key[("193", "HDI")]["Manual_Transcription"] == "godʒijɑ, goʔjkɛd̪ejɑj"
    assert by_key[("193", "SDI")]["Manual_Transcription"] == "gotʃʔ | mɑɾɑo"
    assert by_key[("193", "ORI")]["Manual_Transcription"] == "mɑɾibɑ"


def test_fiftieth_chunk_fly_pair_page_break_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_194)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("194", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("194", "BAI")]["PDF_Page"] == "72"
    assert by_key[("194", "POD")]["PDF_Page"] == "73"
    assert by_key[("194", "CHA")]["Manual_Transcription"] == (
        "ud̪ʌɾenme, ud̪ɑedʒɑnɑe"
    )
    assert by_key[("194", "LAD")]["Manual_Transcription"] == "ɑpiɾme, ɑpiɾdʒɛnɑ"
    assert by_key[("194", "MJH")]["Manual_Transcription"] == "biɾit̪me, ɔtɑŋnenɑj"
    assert by_key[("194", "SDI")]["Manual_Transcription"] == "udɑu | phɑɾkɑo"
    assert by_key[("194", "SNA")]["Manual_Transcription"] == "ud̪oʔpe, ud̪ojenɑj"


def test_fifty_first_chunk_walk_pair_repetitions_source_qualifier_and_blanks():
    rows = guard.load_manual_cells(LEDGER_195)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("195", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("195", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("195", "LAD")]["Manual_Transcription"] == "sɛnodʒɑnʌ, dolɑ"
    assert by_key[("195", "LAD")]["Confidence"] == "medium"
    assert "(?)" in by_key[("195", "LAD")]["Uncertainty"]
    assert by_key[("195", "MUN")]["Manual_Transcription"] == (
        "d̪olɑŋ, sentʃenɑ | d̪olɑŋ, sentʃenɑ"
    )
    assert by_key[("195", "MJH")]["Manual_Transcription"] == (
        "d̪olɑ, senket̪eɾ | d̪olɑ, senket̪eɾ"
    )
    assert by_key[("195", "SDI")]["Manual_Transcription"] == "dɑɾɑ | tɑɾɑm"
    assert by_key[("195", "ORI")]["Manual_Transcription"] == "tʃɑlibɑʔ"


def test_fifty_second_chunk_run_pair_column_break_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_196)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("196", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("196", "HDI")]["Column"] == "left"
    assert by_key[("196", "SDI")]["Column"] == "right"
    assert by_key[("196", "BAI")]["Manual_Transcription"] == (
        "dɦɑud̪em, dɦɑukid̪ɑ"
    )
    assert by_key[("196", "MUN")]["Manual_Transcription"] == "d̪iɾime, niɾkid̪ɑi"
    assert by_key[("196", "MDI")]["Manual_Transcription"] == "niɾ | dɑuɾi"
    assert by_key[("196", "SDI")]["Manual_Transcription"] == "niɾ | dɑɾ"
    assert by_key[("196", "SNA")]["Manual_Transcription"] == "d̪ɛd̪pee, d̪ɛd̪kijɑ"


def test_fifty_third_chunk_go_pair_glottals_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_197)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("197", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("197", "LAD")]["Manual_Transcription"] == "senoʔome, senodʒɑnʌ"
    assert by_key[("197", "MUN")]["Manual_Transcription"] == "d̪olɑ, sindʒenɑ"
    assert by_key[("197", "MCH")]["Manual_Transcription"] == "sen, senoʔodʒɑnɑ"
    assert by_key[("197", "MJH")]["Manual_Transcription"] == "dʒu, senojenɑ"
    assert by_key[("197", "SDI")]["Manual_Transcription"] == "sen | tʃɑlɑkʔ"
    assert by_key[("197", "SNA")]["Manual_Transcription"] == "tʃɛlɑinɑj, tʃɛlɑpe"


def test_fifty_fourth_chunk_come_pair_glottals_length_and_blank():
    rows = guard.load_manual_cells(LEDGER_198)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("198", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("198", "BAI")]["Manual_Transcription"] == "hidʒuʔme, hiːlenɑ"
    assert by_key[("198", "MOH")]["Manual_Transcription"] == "hidʒuʔme, hiːlenɑ"
    assert by_key[("198", "MCH")]["Manual_Transcription"] == (
        "hidʒuʔu, hidʒuʔudʒɑnɑ"
    )
    assert by_key[("198", "MJH")]["Manual_Transcription"] == "hudʒuʔme, hudʒuine"
    assert by_key[("198", "SNA")]["Manual_Transcription"] == "hedʒime, heʔjenɑj"
    assert by_key[("198", "ORI")]["Manual_Transcription"] == "ɑːso"


def test_fifty_fifth_chunk_speak_pair_page_break_palatal_nasal_and_blank():
    rows = guard.load_manual_cells(LEDGER_199)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("199", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("199", "DIG")]["PDF_Page"] == "73"
    assert by_key[("199", "DUM")]["PDF_Page"] == "74"
    assert by_key[("199", "CHA")]["Manual_Transcription"] == (
        "kɑdʒilijɑe | menkejʌe"
    )
    assert by_key[("199", "MOH")]["Manual_Transcription"] == "kɛdʒiʔɲe, kɛdʒilɛʔjɑ"
    assert by_key[("199", "MDI")]["Manual_Transcription"] == "kɑdʒi | men"
    assert by_key[("199", "SDI")]["Manual_Transcription"] == "men | ɾoɾ"
    assert by_key[("199", "ORI")]["Manual_Transcription"] == "kɔhilɑ, kuhɑ"


def test_fifty_sixth_chunk_listen_pair_continuations_dentals_and_blank():
    rows = guard.load_manual_cells(LEDGER_200)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("200", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("200", "CHA")]["Manual_Transcription"] == (
        "ɑjuməjɑi, ɑjumkɛdɑʔi | ɑjumem, ɑjumlijɑe"
    )
    assert by_key[("200", "MOH")]["Manual_Transcription"] == "ɑjɑmt̪ɑ, əjɑmkejɑ"
    assert by_key[("200", "MUN")]["Manual_Transcription"] == "ɑjomet̪ɑnɑ, ɑjumked̪ɑi"
    assert by_key[("200", "MCH")]["Manual_Transcription"] == (
        "ɑjumem, ɑjumkidɑ | ɑjum, ɑjumkidɑʔɑ"
    )
    assert by_key[("200", "MJH")]["Manual_Transcription"] == (
        "ɑjumem, ɑjumkidɑ | ɑjumt̪enɑj, ɑjumked̪e"
    )
    assert by_key[("200", "ORI")]["Manual_Transcription"] == "suno"


def test_fifty_seventh_chunk_look_pair_column_break_length_palatal_and_blank():
    rows = guard.load_manual_cells(LEDGER_201)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("201", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("201", "POD")]["Column"] == "left"
    assert by_key[("201", "UDA")]["Column"] == "right"
    assert by_key[("201", "LAD")]["Manual_Transcription"] == "lelːime, lelːid̪ɑ"
    assert by_key[("201", "MUN")]["Manual_Transcription"] == "nelt̪enɑ, ɑinelkid̪ɑi"
    assert by_key[("201", "MJH")]["Manual_Transcription"] == "lɛlɛjeʔ, lɛlket̪e"
    assert by_key[("201", "SNA")]["Manual_Transcription"] == "ɲɛɲelkɛnɑj, ɲelkijɑj"
    assert by_key[("201", "ORI")]["Manual_Transcription"] == "dekho"


def test_fifty_eighth_chunk_first_person_nasals_repetitions_and_blank():
    rows = guard.load_manual_cells(LEDGER_202)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("202", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("202", "BAI")]["Manual_Transcription"] == "ɑŋ"
    assert by_key[("202", "UDA")]["Manual_Transcription"] == "ɑn"
    assert by_key[("202", "MCH")]["Manual_Transcription"] == "ɑiŋ | ɑiŋ"
    assert by_key[("202", "MDI")]["Manual_Transcription"] == "ɑiŋ | iŋ | ɑiŋ"
    assert by_key[("202", "SDI")]["Manual_Transcription"] == "in"
    assert by_key[("202", "SNA")]["Manual_Transcription"] == "iŋ"
    assert by_key[("202", "ORI")]["Manual_Transcription"] == "mũ"


def test_fifty_ninth_chunk_second_person_length_dental_and_blank():
    rows = guard.load_manual_cells(LEDGER_203)
    assert len(rows) == 18
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("203", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("203", "LAD")]["Manual_Transcription"] == "ɑmː"
    assert by_key[("203", "BAI")]["Manual_Transcription"] == "ɑm"
    assert by_key[("203", "SNA")]["Manual_Transcription"] == "ɑm"
    assert by_key[("203", "ORI")]["Manual_Transcription"] == "t̪u"


def test_sixtieth_chunk_pronouns_blanks_length_glottals_and_variants():
    rows = guard.load_manual_cells(LEDGER_204_208)
    assert len(rows) == 90
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert all(by_key[(str(item), "DIG")]["Review_Status"] == "source_blank"
               for item in range(204, 209))
    assert by_key[("204", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("206", "MDI")]["Review_Status"] == "source_blank"
    assert by_key[("204", "LAD")]["Manual_Transcription"] == "ɑbɛn"
    assert by_key[("204", "SNA")]["Manual_Transcription"] == "ɑbiŋ"
    assert by_key[("205", "CHA")]["Manual_Transcription"] == "ɑʔt̪"
    assert by_key[("206", "CHA")]["Manual_Transcription"] == "ɑtʔ"
    assert by_key[("206", "MCH")]["Column"] == "left"
    assert by_key[("206", "MDI")]["Column"] == "right"
    assert by_key[("207", "ORI")]["Manual_Transcription"] == "ɑme | ɑmpe"
    assert by_key[("208", "LAD")]["Manual_Transcription"] == "ɑlːe"
    assert by_key[("208", "ORI")]["Source_Cognate_Labels"] == "2 | 2"


def test_sixty_first_chunk_final_pronouns_continuations_and_length():
    rows = guard.load_manual_cells(LEDGER_209_210)
    assert len(rows) == 36
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("209", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("210", "DIG")]["Review_Status"] == "source_blank"
    assert by_key[("209", "LAD")]["Manual_Transcription"] == "ɑpeʔ"
    assert by_key[("209", "MUN")]["Manual_Transcription"] == "ɑŋku"
    assert by_key[("209", "MDI")]["Manual_Transcription"] == "ɑpeɑ"
    assert by_key[("210", "MDH")]["Manual_Transcription"] == "ɑko | ɑko"
    assert by_key[("210", "SNA")]["Manual_Transcription"] == "uŋkin | uŋkuʔko"
    assert by_key[("210", "ORI")]["Manual_Transcription"] == "se mɑnːe"
    assert by_key[("210", "MDH")]["Source_Cognate_Labels"] == "1 | 1"


def test_ho_2024_overlap_is_same_five_elicitations_and_status_complete():
    rows = overlap.build_rows()
    assert len(rows) == 5 * 210 == 1050
    assert len({row["Durable_List_ID"] for row in rows}) == 5
    assert len({row["Durable_Cell_ID"] for row in rows}) == 1050
    assert {row["Status_Parity"] for row in rows} == {"yes"}
    assert sum(row["Representation_Comparison"] == "blank-parity" for row in rows) == 11
    assert sum(
        row["Representation_Comparison"] == "unicode-exact-after-label-removal"
        for row in rows
    ) == 221
    assert sum(
        row["Representation_Comparison"] == "publication-transcription-differs"
        for row in rows
    ) == 818
    assert {row["Canonical_Publication"] for row in rows} == {
        "baileymaggard2015bhumij"
    }


def test_registry_staging_audit_and_durable_entry_keys_are_exhaustive():
    registry = guard.load_registry()
    assert len(registry) == 18
    assert {code for code, row in registry.items() if row["Install"] == "yes"} == {
        "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"
    }
    assert registry["BAI"]["Dialect_ID"] == "bhumij1989-baigodia"
    assert registry["POD"]["Ho2024_Overlap_Code"] == "BMA"
    assert registry["UDA"]["Source_Language_Label"] == "Mundari? Bhumij?"
    rows = guard.load_manual_ledgers(LEDGERS)
    forms = guard.stage_target_forms(rows, registry)
    audit = guard.build_audit(rows, registry)
    assert len(forms) == 2100
    assert len(audit) == 3780
    assert len({row["Entry_Key"] for row in forms}) == 2100
    assert sum(row["Disposition"] == "target-staged" for row in audit) == 2054
    assert sum(row["Disposition"] == "target-source-blank-excluded" for row in audit) == 46
    assert sum(row["Disposition"] == "comparison-control-excluded" for row in audit) == 1636
    assert sum(
        row["Disposition"] == "comparison-control-blank-excluded" for row in audit
    ) == 44
    assert all(row["Language_ID"] == "unr" for row in forms)
    assert all(row["Source"].startswith("baileymaggard2015bhumij[") for row in forms)


def test_sil_bhumij_profile_covers_every_staged_form_and_key_symbols():
    rows = guard.load_manual_ledgers(LEDGERS)
    forms = guard.stage_target_forms(rows)
    inventory, unmatched = guard.profile_inventory(forms)
    assert len(inventory) == 53
    assert not unmatched
    assert {row["Covered_By_Profile"] for row in inventory} == {"yes"}
    profile = Tokenizer(str(ROOT / "conversion/sil-bhumij.txt"))

    def convert(value):
        return unicodedata.normalize(
            "NFC", profile(unicodedata.normalize("NFC", value), column="IPA")
            .replace(" ", "")
            .replace("#", " ")
        )

    assert convert("ɑpeʔ") == "āpeʔ"
    assert convert("d̪ɑtɑ") == "dātā"
    assert convert("se mɑnːe") == "se mānːe"
    assert convert("metʔn̩") == "metʔn̩"
    assert all("�" not in convert(row["Form"]) for row in forms)
