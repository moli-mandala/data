import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-magar-2024.csv"


def test_magar_2024_clean_pdf_targets_are_complete():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1769
    assert all(len(row) == 11 and row[2] == row[5] for row in rows)
    assert Counter(row[0] for row in rows) == {
        "western_magar_lasargha": 272,
        "western_magar_mathagadhi": 287,
        "western_magar_jhokedi": 44,
        "central_magar_siluwa": 46,
        "central_magar_mityal": 310,
        "central_magar_dhardh": 41,
        "central_magar_inaskot": 46,
        "central_magar_michhurlung": 44,
        "central_magar_rhising": 288,
        "central_magar_arkhala": 305,
        "central_magar_raikot": 43,
        "central_magar_bhadari": 43,
    }
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(
        range(60, 78)
    )
    assert len({row[10] for row in rows}) == 1769
    assert all(not row[0].startswith("eastern_magar_") for row in rows)
    assert all(row[2] not in {"", "N", "DK", "--", "Nepali"} for row in rows)
    assert all("�" not in row[2] for row in rows)


def test_magar_2024_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"WesternMagar"}
    assert {dialects[lect]["Glottocode"] for lect in lects} == {"west2418"}


def test_magar_2024_ipa_profile():
    profile = Tokenizer("conversion/magar-2024.txt")
    cases = {
        "dʒjæn": "jyæn",
        "midzjan": "miʣyan",
        "mitʃʰam": "micʰam",
        "mita mitãhã̤": "mita mitãhã̤",
        "mik̚": "mik̚",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize(
            "NFC", converted.replace(" ", "").replace("#", " ")
        )
        assert converted == expected


def test_magar_2024_compiled_rows_keep_source_entries():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("magar-2024:")
        ]
    assert len(rows) == 1769
    assert len({row["Source_Key"] for row in rows}) == 1769
