import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-yamphu.csv"


def test_yamphu_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 2188
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("hilty-mitchell2014[p. ") for row in rows)
    assert all(row[10].startswith("yamphu:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(109, 130))


def test_yamphu_import_keeps_all_nine_site_lects():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "lohorung_angala": 240,
        "lohorung_dhupu": 200,
        "lohorung_gairi_pangma": 251,
        "southern_yamphu_devitar": 246,
        "southern_yamphu_rajarani": 245,
        "yamphu_hedangna": 255,
        "yamphu_khoktak": 241,
        "yamphu_num": 264,
        "yamphu_seduwa": 246,
    }


def test_yamphu_lects_are_registered_under_report_language_groups():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {
        "Lohorung",
        "SouthernYamphu",
        "Yamphu",
    }
    assert dialects["southern_yamphu_rajarani"]["Language_ID"] == "SouthernYamphu"


def test_yamphu_ipa_profile():
    profile = Tokenizer("conversion/yamphu.txt")
    cases = {
        "lɪnʈa": "linṭa",
        "dʒʌɾa": "jara",
        "tsʌŋak̚": "ʦaŋak̚",
        "kʰʌɹani": "kʰarani",
        "ɠʷɛŋ": "gʷeŋ",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
