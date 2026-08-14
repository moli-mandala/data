import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-magahi.csv"


def test_magahi_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1050
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("thakur-thakur2016magahi[p. ") for row in rows)
    assert all(row[10].startswith("magahi:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(113, 122))
    assert {int(row[10].split(":")[1]) for row in rows} == set(range(1, 211))


def test_magahi_import_keeps_only_the_five_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "magahi_sarlahi": 210,
        "magahi_mahottari": 210,
        "magahi_dhanusha": 210,
        "magahi_saptari": 210,
        "magahi_morang": 210,
    }


def test_magahi_lects_follow_current_glottolog_classification():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}

    keys = {key for key in dialects if key.startswith("magahi_")}
    assert keys == {
        "magahi_sarlahi",
        "magahi_mahottari",
        "magahi_dhanusha",
        "magahi_saptari",
        "magahi_morang",
    }
    assert {dialects[key]["Language_ID"] for key in keys} == {"MagahiNepal"}
    assert {dialects[key]["Glottocode"] for key in keys} == {"maga1260"}


def test_magahi_profile_preserves_source_ipa_separately():
    profile = Tokenizer("conversion/magahi-survey.txt")
    cases = {
        "kəpar": "kapār",
        "jʰãɽa": "jʰā̃ṛā",
        "həd̺d̺i": "haddī",
        "ciniyã bədam": "cīnīyā̃ badām",
        "pʰut̺əl": "pʰūtal",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
