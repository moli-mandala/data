import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-naaba.csv"


def test_naaba_clean_pdf_targets_are_complete():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 665
    assert all(len(row) == 11 and row[2] == row[5] for row in rows)
    assert Counter(row[0] for row in rows) == {
        "naaba_pibu": 340,
        "naaba_kimathanka": 325,
    }
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(67, 82))
    assert len({row[10] for row in rows}) == 665
    assert all(row[2] not in {"No entry", "Nepali"} for row in rows)
    assert all("Њ" not in row[2] and "�" not in row[2] for row in rows)


def test_naaba_ipa_profile():
    profile = Tokenizer("conversion/naaba.txt")
    cases = {
        "tɕʰeraŋ": "ʦ̣ʰeraŋ",
        "ɡo̤": "go̤",
        "bĩ̤zo": "bĩ̤zo",
        "(pʌrtʃʰʌ)": "partśʰa",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected


def test_naaba_compiled_rows_keep_source_entries():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["Source_Key"].startswith("naaba:")]
    assert len(rows) == 665
    assert len({row["Source_Key"] for row in rows}) == 665
