import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-kurux-nepal.csv"


def test_kurux_nepal_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1336
    assert Counter(row[0] for row in rows) == {
        "kurux_lochani": 333,
        "kurux_bhokraha": 332,
        "kurux_siddhapur": 334,
        "kurux_tokla": 337,
    }
    assert all(len(row) == 11 for row in rows)
    assert all(row[7].startswith("shackelford-swenson-chaudhary-maggard2022kurux[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(44, 57))
    assert len({row[10] for row in rows}) == len(rows)
    assert len({int(row[10].split(":")[1]) for row in rows}) == 329
    assert not any("�" in row[2] for row in rows)


def test_kurux_nepal_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Kurux"}


def test_kurux_nepal_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/kurux-nepal.txt")
    cases = {
        "t͡ʃʰuʈʈi": "cʰuṭṭi",
        "d͡ʒʰia": "jʰia",
        "χɛ̃s": "xẽs",
        "pʌĩ̯ja": "paĩ̯ya",
        "kiɽa lʌgia": "kiṛa#lagia",
        "bṳŋgiʌs": "bṳŋgias",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", ""))
        assert converted == expected
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
