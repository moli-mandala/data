import csv
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-kudiya.csv"


def test_kudiya_import_is_complete_and_target_only():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 409
    assert all(len(row) == 11 for row in rows)
    assert Counter(row[0] for row in rows) == {"kudiya_g1": 205, "kudiya_k1": 204}
    assert all(row[7].startswith("joseph2024kudiya[p. ") for row in rows)
    assert len({row[10] for row in rows}) == 409


def test_kudiya_lects_are_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    assert dialects["kudiya_g1"]["Language_ID"] == "Kudiya"
    assert dialects["kudiya_k1"]["Language_ID"] == "Kudiya"


def test_kudiya_profile_covers_every_source_form():
    profile = Tokenizer(DATA_DIR / "conversion/kudiya.txt")
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in profile(row[2], column="IPA")
