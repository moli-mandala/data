import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-tagin-puroik.csv"


def test_tagin_puroik_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 4939
    assert all(len(row) == 11 for row in rows)
    assert all(row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("abraham-sako2021[p. ") for row in rows)
    assert all(row[10].startswith("tagin-puroik:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)


def test_tagin_puroik_import_keeps_all_sixteen_lects():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "puroik_phereng": 309,
        "puroik_gari": 299,
        "puroik_chug": 303,
        "puroik_paji": 295,
        "bugun_singchung": 309,
        "bugun_wangho": 316,
        "bugun_bichom": 345,
        "bugun_kaspi": 310,
        "bugun_namphri": 319,
        "tagin_sippi": 311,
        "tagin_nacho": 305,
        "tagin_baki": 313,
        "tagin_taliha": 308,
        "tagin_maskia": 292,
        "tagin_takseng": 293,
        "nyishi_chimpu": 312,
    }


def test_tagin_puroik_lects_are_registered_as_dialects():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        lects = {row[0] for row in csv.reader(stream)}

    assert lects <= dialects.keys()
    assert {dialects[lect]["Language_ID"] for lect in lects} == {
        "Bugun",
        "Nyishi",
        "Puroik",
        "Tagin",
    }


def test_tagin_puroik_ipa_profile():
    profile = Tokenizer("conversion/tagin-puroik.txt")
    cases = {
        "t̪ʌɾop": "tarop",
        "pat̪ːa": "patta",
        "ɲed̪okolo": "ñedokolo",
        "tʃeɲiad̪oɲia": "ceñiadoñia",
        "kəgtsiŋ": "kəgʦiŋ",
        "pharamphakhuŋ": "pʰarampʰakʰuŋ",
        "ʃĕandəkhau": "śeandəkʰau",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFD", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
