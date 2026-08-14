import csv
import unicodedata
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA_DIR = Path(__file__).parents[1]
SOURCE_FILE = DATA_DIR / "data/other/forms/20260813-pyangaun-newar.csv"


def test_pyangaun_import_is_complete_and_target_only():
    with SOURCE_FILE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 338
    assert all(len(row) == 11 for row in rows)
    assert {row[0] for row in rows} == {"pyangaun_newar"}
    assert all(row[7].startswith("smith2021pyangaun[p. ") for row in rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(32, 45))
    assert len({row[10] for row in rows}) == len(rows)
    assert not any("�" in row[2] for row in rows)
    source_items = {int(row[10].split(":")[1]) for row in rows}
    assert set(range(1, 326)) - source_items == {85, 86, 91, 92, 95, 136, 193}


def test_pyangaun_lect_is_registered():
    with (DATA_DIR / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    assert dialects["pyangaun_newar"]["Language_ID"] == "New"


def test_pyangaun_profile_converts_ipa_and_covers_every_form():
    profile = Tokenizer(DATA_DIR / "conversion/pyangaun-newar.txt")
    cases = {
        "d͡zi": "ʣi",
        "t͡sʰə̃ŋ": "ʦʰə̃ŋ",
        "miːkʰa": "mīkʰa",
        "t͡ʃəlpʰəl jætə": "cəlpʰəl yætə",
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
