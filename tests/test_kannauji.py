import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20230526-kannauji.csv"


def test_kannauji_partial_is_completed_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 3033
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("kannauji[p. ") for row in rows)
    assert all(row[10].startswith("kannauji:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(55, 108))


def test_kannauji_keeps_targets_and_excludes_controls_and_nonforms():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    counts = Counter(row[0] for row in rows)

    assert counts == {
        "Dehati-Badeli": 218,
        "Dehati-Kirkkichiyapur": 235,
        "Dehati-Madnapur": 226,
        "Dehati-Sikandarpur": 238,
        "Hindi-Dhubar": 232,
        "Hindi-Gabchariyapur": 230,
        "Hindi-Gohaniya": 228,
        "Hindi-Jamniya": 244,
        "Hindi-Rohili": 244,
        "Hindi-Sanayak": 232,
        "Hindi-Sarhati": 238,
        "Hindi-Saraiyya": 237,
        "Kannauji-Central": 231,
    }
    assert not ({"Hindi", "Bundeli", "Braj Bhasha"} & counts.keys())
    assert all("ɴɑɱɛ" not in row[2] and "βʏ" not in row[2] for row in rows)


def test_kannauji_ipa_profile():
    profile = Tokenizer("conversion/kannauji.txt")
    cases = {
        "d̪ʌɾʋʌdʒʌ": "darvaja",
        "tʃhoʈobɦʌija": "choṭobʰaiya",
        "d̪ɦʊã": "dʰuã",
        "mʊh̰": "muh̰",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected


def test_kannauji_compiled_rows_keep_distinct_source_entries():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("kannauji:")
        ]

    assert len(rows) == 3033
    assert len({row["Source_Key"] for row in rows}) == 3033
