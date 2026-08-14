import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-dewas-rai.csv"


def _convert(value: str) -> str:
    profile = Tokenizer("conversion/dewas-rai.txt")
    return unicodedata.normalize(
        "NFC",
        profile(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_dewas_rai_clean_pdf_targets_are_complete():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 1922
    assert all(len(row) == 11 and row[2] == row[5] for row in rows)
    assert Counter(row[0] for row in rows) == {
        "dewas_rai_mahendra_jhyadi": 323,
        "dewas_rai_singoul": 319,
        "dewas_rai_majhgaun": 319,
        "done_danuwar_jaretar": 321,
        "danuwar_chandanpur": 319,
        "kochariya_singoul": 321,
    }
    assert len({row[10] for row in rows}) == 1922
    assert {int(row[10].split(":")[1]) for row in rows} == set(range(1, 326)) - {213, 215}
    assert all("Nepali" not in row[0] and "�" not in row[2] for row in rows)
    assert all(row[2] not in {"", "-", "A", "B", "C", "D", "x"} for row in rows)
    assert all(row[2] == " ".join(t for t in row[2].split() if t not in {"A", "B", "C", "D", "x"}) for row in rows)


def test_dewas_rai_ipa_profile_and_full_source_coverage():
    assert _convert("bʰʌ̃ĩ̯si") == "bʰãĩ̯si"
    assert _convert("dʒungʰuɣau") == "jungʰuɣau"
    assert _convert("pias lagʎʌ") == "pias lagʎa"
    # NFC recomposes the profile's u + breathy-voice mark as U+1E73.
    assert _convert("mṳi") == "mṳi"
    assert _convert("tʃʰɑlkʌ") == "cʰālka"

    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in _convert(row[2])


def test_dewas_rai_lects_are_registered_with_catalog_groups():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}

    expected = {
        "dewas_rai_mahendra_jhyadi": ("DewasDoneDanuwar", "dhan1265"),
        "dewas_rai_singoul": ("DewasDoneDanuwar", "dhan1265"),
        "dewas_rai_majhgaun": ("DewasDoneDanuwar", "dhan1265"),
        "done_danuwar_jaretar": ("DewasDoneDanuwar", "dhan1265"),
        "danuwar_chandanpur": ("KochariyaEastDanuwar", "koch1253"),
        "kochariya_singoul": ("KochariyaEastDanuwar", "koch1253"),
    }
    assert {key: (dialects[key]["Language_ID"], dialects[key]["Glottocode"]) for key in expected} == expected


def test_dewas_rai_compiled_rows_keep_raw_source_transcription():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        source_keys = [
            row
            for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("dewas-rai:")
        ]
    assert len(source_keys) == 1922
    assert len({row["Source_Key"] for row in source_keys}) == 1922

    legacy_ids = {row["Legacy_ID"] for row in source_keys}
    with open("cldf/form-id-aliases.csv", encoding="utf-8", newline="") as stream:
        aliases = {
            row["Legacy_ID"]: row["Form_ID"]
            for row in csv.DictReader(stream)
            if row["Legacy_ID"] in legacy_ids
        }
    assert aliases.keys() == legacy_ids
    form_ids = set(aliases.values())
    with open("cldf/forms.csv", encoding="utf-8", newline="") as stream:
        forms = [row for row in csv.DictReader(stream) if row["ID"] in form_ids]
    assert len(forms) == 1922
    assert all(row["Phonemic"] for row in forms)
    assert sum(row["Form"] != row["Phonemic"] for row in forms) > 1000
