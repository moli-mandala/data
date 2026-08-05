import csv
import io
import unicodedata
import sys
from pathlib import Path

from segments import Tokenizer

DATA_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(DATA_DIR))

from make_cldf import SCHMIDT_PROFILE_LANGUAGES, normalize_schmidt_stress, parse_file


def convert(source: str) -> str:
    profile = Tokenizer("conversion/schmidt-kashmiri.txt")
    prepared = normalize_schmidt_stress(unicodedata.normalize("NFC", source))
    result = profile(prepared, column="IPA").replace(" ", "").replace("#", " ")
    return unicodedata.normalize("NFC", result)


def test_schmidt_kashmiri_profile_and_language_routing():
    assert SCHMIDT_PROFILE_LANGUAGES == {"K", "kash", "pog", "sir"}
    assert convert("də:ṛ") == "də̄ṛ"


def test_schmidt_poguli_length_palatalisation_and_stress():
    assert convert("nʲu:l as'ma:n") == "nʸūl asmā́n"


def test_schmidt_sarazi_length_and_noninitial_stress():
    assert convert("kasa:'li:") == "kasālī́"
    assert convert("džama'tro:") == "jamatrṓ"


def test_all_schmidt_table_3_languages_are_routed_through_profile(tmp_path):
    source = tmp_path / "other" / "forms" / "schmidt.csv"
    source.parent.mkdir(parents=True)
    rows = [
        ["K", "1", "də:ṛ", "beard", "", "də:ṛ", "", "schmidt"],
        ["pog", "1", "nʲu:l", "blue", "", "nʲu:l", "", "schmidt"],
        ["sir", "1", "kasa:'li:", "armpit", "", "kasa:'li:", "", "schmidt"],
        ["kash", "1", "tə̌:ṛi", "beard", "", "tə̌:ṛi", "", "schmidt"],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    parsed, stats = parse_file(str(source), io.StringIO())

    assert [row.form for row in parsed] == ["də̄ṛ", "nʸūl", "kasālī́", "tə̄̌ṛi"]
    assert [row.ipa for row in parsed] == [row[5] for row in rows]
    assert stats == {"converted": 4, "for_conversion": 4}
