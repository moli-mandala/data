import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-mewahang.csv"


def test_mewahang_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1164
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("rai-rai-thokar2014mewahang[p. ") for row in rows)
    assert all(row[10].startswith("mewahang:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(122, 128))
    assert {int(row[10].split(":")[1]) for row in rows} == (
        set(range(1, 211)) - {73, 176, 210}
    )


def test_mewahang_import_keeps_only_the_five_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "eastern_mewahang_yaphu": 234,
        "eastern_mewahang_mangtewa": 222,
        "western_mewahang_tamku": 237,
        "western_mewahang_bala": 230,
        "western_mewahang_yamdang": 241,
    }


def test_mewahang_lects_follow_the_current_eastern_western_split():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}

    assert dialects["eastern_mewahang_yaphu"]["Language_ID"] == "EasternMewahang"
    assert dialects["eastern_mewahang_mangtewa"]["Language_ID"] == "EasternMewahang"
    assert dialects["western_mewahang_tamku"]["Language_ID"] == "WesternMewahang"
    assert dialects["western_mewahang_bala"]["Language_ID"] == "WesternMewahang"
    assert dialects["western_mewahang_yamdang"]["Language_ID"] == "WesternMewahang"
    assert {dialects[key]["Glottocode"] for key in dialects if "mewahang_" in key} == {
        "east2357", "west2422"
    }


def test_mewahang_profile_preserves_source_ipa_separately():
    profile = Tokenizer("conversion/mewahang.txt")
    cases = {
        "tˢʰebruŋwa": "ʦʰebruŋva",
        "mimtᶳʰa": "mimcʰa",
        "sumdᶻi": "sumʣi",
        "pɨ:ʔma": "pɨ̄ʔma",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
