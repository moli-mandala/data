import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-dotyali.csv"


def test_dotyali_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1270
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(
        row[7].startswith("eichentopf-tupper2019dotyali[p. ") for row in rows
    )
    assert all(row[10].startswith("dotyali:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(28, 41))
    assert len({row[10].split(":")[1] for row in rows}) == 311


def test_dotyali_import_keeps_only_the_four_target_varieties():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "dotyali_doti": 316,
        "dotyali_baitadi": 319,
        "dotyali_darchula": 318,
        "dotyali_bajhang": 317,
    }


def test_dotyali_lects_are_registered_to_dotyali():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Dotyali"}


def test_dotyali_ipa_profile():
    profile = Tokenizer("conversion/dotyali.txt")
    cases = {
        "həɾ": "hər",
        "səriʒ": "səriž",
        "tʃʰɑti": "cʰāti",
        "kɑ̃ɖɑ̃": "kā̃ḍā̃",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
