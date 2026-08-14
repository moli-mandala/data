import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-chhulung.csv"


def test_chhulung_import_is_complete_and_page_cited():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 970
    assert all(len(row) == 11 for row in rows)
    assert all(row[0] and row[2] and row[3] and row[5] for row in rows)
    assert all(row[7].startswith("rai-rai-thokar2014chhulung[p. ") for row in rows)
    assert all(row[10].startswith("chhulung:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert {int(row[7].split("p. ")[1][:-1]) for row in rows} == set(range(116, 123))
    assert {int(row[10].split(":")[1]) for row in rows} == (
        set(range(1, 195)) - {18}
    )


def test_chhulung_import_keeps_only_the_five_target_sites():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        counts = Counter(row[0] for row in csv.reader(stream))

    assert counts == {
        "chhulung_barbhanjyang": 194,
        "chhulung_gairi": 194,
        "chhulung_pakha": 194,
        "chhulung_pokla": 194,
        "chhulung_suke_ahal": 194,
    }


def test_chhulung_lects_follow_current_glottolog_classification():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}

    keys = {key for key in dialects if key.startswith("chhulung_")}
    assert keys == {
        "chhulung_barbhanjyang",
        "chhulung_gairi",
        "chhulung_pakha",
        "chhulung_pokla",
        "chhulung_suke_ahal",
    }
    assert {dialects[key]["Language_ID"] for key in keys} == {"Chhulung"}
    assert {dialects[key]["Glottocode"] for key in keys} == {"chhu1238"}


def test_chhulung_profile_preserves_source_ipa_separately():
    profile = Tokenizer("conversion/chhulung.txt")
    cases = {
        "dzʰarak": "ʣʰarak",
        "ŋa?lasi": "ŋaʔlasi",
        "Muk": "muk",
        "Jam": "yam",
        "hərd̪i": "hərdi",
        "heɾe": "here",
    }
    for source, expected in cases.items():
        converted = profile(unicodedata.normalize("NFC", source), column="IPA")
        converted = unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))
        assert converted == expected
