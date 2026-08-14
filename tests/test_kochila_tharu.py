import csv
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-kochila-tharu.csv"


def test_kochila_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 833
    assert all(len(row) == 11 for row in rows)
    assert Counter(row[0] for row in rows) == {
        "kochila_morang_east": 279,
        "kochila_bara_west": 274,
        "kochila_siraha_central": 280,
    }
    assert all(row[7].startswith("eichentopf-mitchell2020kochila[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(24, 59))
    assert len({row[10] for row in rows}) == len(rows)
    assert not any("�" in row[2] for row in rows)


def test_kochila_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}
    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {"KochilaTharu"}


def test_kochila_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/kochila-tharu.txt")
    cases = {
        "dʒɔʈə": "joṭə",
        "kʰẽⁱs": "kʰẽⁱs",
        "tʃaⁱ̃t̪i": "caⁱ̃ti",
        "a:ŋ": "āŋ",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize(
            "NFC", converted.replace(" ", "").replace("#", " ")
        )
        assert converted == expected
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
