import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-maikoti-kham.csv"


def test_maikoti_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1147
    assert Counter(row[0] for row in rows) == {
        "maikoti_maikot": 290,
        "maikoti_ranma": 284,
        "maikoti_arjal": 287,
        "maikoti_hukam": 286,
    }
    assert all(len(row) == 11 for row in rows)
    assert all(row[7].startswith("leman2020maikoti[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(38, 51))
    assert len({row[10] for row in rows}) == len(rows)
    assert not any("�" in row[2] for row in rows)


def test_maikoti_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Kjl"}


def test_maikoti_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/maikoti-kham.txt")
    cases = {
        "t͡sɛm": "ʦem",
        "d͡zimi": "ʣimi",
        "t͡ʃĩ": "cĩ",
        "kʷãkʰãnã": "kʷãkʰãnã",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
