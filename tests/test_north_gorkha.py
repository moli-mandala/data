import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-north-gorkha.csv"


def _rows():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_north_gorkha_import_is_complete_and_target_only():
    rows = _rows()
    assert len(rows) == 4269
    assert Counter(row[0] for row in rows) == {
        "nubri_sama": 225,
        "nubri_lho": 222,
        "nubri_namrung": 231,
        "nubri_prok": 271,
        "tsum_chekampar": 232,
        "southern_ghale_barpak": 235,
        "southern_ghale_kyaura": 229,
        "southern_ghale_laprak": 229,
        "northern_ghale_jagat": 238,
        "northern_ghale_philim": 236,
        "northern_ghale_uiya": 235,
        "northern_ghale_khorla": 230,
        "northern_ghale_nyak": 236,
        "kutang_ghale_bihi": 213,
        "kutang_ghale_chyak": 216,
        "kutang_ghale_rana": 129,
        "eastern_gorkha_tamang_kashigaun": 221,
        "eastern_gorkha_tamang_keraunja": 215,
        "western_tamang_lamagara": 226,
    }
    assert all(len(row) == 11 for row in rows)
    assert all(row[7].startswith("webster2022north-gorkha[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(25, 70))
    assert len({row[10] for row in rows}) == len(rows)
    concepts = {
        section: {int(row[10].split(":")[2]) for row in rows if row[10].split(":")[1] == section}
        for section in ("tibetan", "gorkha")
    }
    assert concepts == {
        "tibetan": set(range(1, 240)) - {67, 68, 74, 81, 98},
        "gorkha": set(range(1, 242)) - {73, 74, 174, 177, 204, 206, 230, 231, 232, 233, 235, 237},
    }
    assert {row[3] for row in rows if ":gorkha:203:" in row[10]} == {"you (sg. informal)"}
    assert not any("�" in row[2] for row in rows)


def test_north_gorkha_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    lects = {row[0] for row in _rows()}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {
        "Nubri", "Tsum", "SouthernGhale", "NorthernGhale", "KutangGhale",
        "EasternGorkhaTamang", "WesternTamang",
    }


def test_north_gorkha_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/north-gorkha.txt")
    cases = {
        "tʃʰee": "cʰee",
        "dʒɛ": "je",
        "ʈʰɹak": "ṭʰrak",
        "n̪a": "n̪a",
        "li̤:": "li̤ː",
        "ki lo": "ki#lo",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", ""))
        assert converted == expected
    for row in _rows():
        assert "�" not in profile(unicodedata.normalize("NFC", row[2]), column="IPA")
