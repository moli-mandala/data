import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-mustang-loke.csv"


def test_mustang_loke_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1996
    assert Counter(row[0] for row in rows) == {
        "loke_lo_manthang": 437,
        "loke_ghiling": 372,
        "loke_chhosher": 396,
        "loke_jharkot": 389,
        "loke_kagbeni": 402,
    }
    assert all(len(row) == 11 for row in rows)
    assert all(row[7].startswith("khadgi-marcuson-marcuson2021mustang[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(126, 162))
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[10].split(":")[1]) for row in rows} == set(range(1, 337))
    assert not any("�" in row[2] or " " in row[2] for row in rows)


def test_mustang_loke_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"Loy"}


def test_mustang_loke_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/mustang-loke.txt")
    cases = {
        "sṳwu": "sṳvu",
        "tɕʰi̤mba": "ʦ̣ʰi̤mba",
        "rœ̤tœ̤ʔ": "rö̤tö̤ʔ",
        "ɲæ̤l-": "ñæ̤l-",
        "ɦo̤ː-": "hō̤-",
        "(tsʰa)pø̤tɕa": "(ʦʰa)pö̤ʦ̣a",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", ""))
        assert converted == expected
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
