import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-majhi-bote.csv"


def test_majhi_bote_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1590
    assert all(len(row) == 11 for row in rows)
    assert Counter(row[0] for row in rows) == {
        "majhi_kunauri": 264,
        "majhi_gaikura": 269,
        "majhi_majhigau": 268,
        "majhi_pachuwar": 264,
        "bote_kawasoti": 264,
        "bote_madi": 261,
    }
    assert all(row[7].startswith("page2024majhi-bote[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(41, 63))
    assert len({row[10] for row in rows}) == len(rows)


def test_majhi_bote_lects_are_registered_to_the_correct_language():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert {dialects[lect]["Language_ID"] for lect in lects if lect.startswith("majhi_")} == {"Majhi"}
    assert {dialects[lect]["Language_ID"] for lect in lects if lect.startswith("bote_")} == {"Bote"}


def test_majhi_bote_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/majhi-bote.txt")
    cases = {
        "tʃʰati": "cʰati",
        "dʒʲu": "jʸu",
        "bɛʈɛk": "beṭek",
        "ʌnaɾ": "anar",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", ""))
        assert converted == expected
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
