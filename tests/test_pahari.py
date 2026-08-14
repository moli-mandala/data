import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-pahari.csv"


def test_pahari_clean_pdf_wordlists_are_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1452
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("smith2022pahari[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(39, 86))
    assert len({row[10] for row in rows}) == len(rows)


def test_pahari_keeps_five_targets_and_excludes_newar_controls_and_nonforms():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    counts = Counter(row[0] for row in rows)

    assert counts == {
        "pahari_jamune": 284,
        "pahari_maasdada": 330,
        "pahari_sakhatar": 212,
        "pahari_salintar": 316,
        "pahari_sikharpa": 310,
    }
    assert not {
        "Kathmandu",
        "Patan",
        "Bhaktapur",
        "Pyangaun",
        "Balami",
        "Chitlang",
        "Dolakha",
    } & counts.keys()
    assert all(row[2] not in {"(N)", "(H)", "DK", "DC", "NA", "No record"} for row in rows)
    assert all("ν" not in row[2] and "�" not in row[2] for row in rows)
    assert any(row[3] == "come down (3S-PT)" for row in rows)
    assert any(row[3] == "he/she (formal sing)" for row in rows)


def test_pahari_ipa_profile():
    profile = Tokenizer("conversion/pahari.txt")
    cases = {
        "m̤ɔ": "m̤o",
        "t͡sʰe": "ʦʰe",
        "d͡z̤i": "ʣ̤i",
        "ɪndɾɛni+pəɾe+du+d͡zu": "indreni+pəre+du+ʣu",
        "t͡saĩ̯.ɦa": "ʦaĩ̯.ɦa",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected


def test_pahari_compiled_rows_keep_distinct_source_entries():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("pahari:")
        ]

    assert len(rows) == 1452
    assert len({row["Source_Key"] for row in rows}) == 1452
