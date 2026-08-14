import csv
import unicodedata
from collections import Counter

from segments.tokenizer import Tokenizer


SOURCE_FILE = "data/other/forms/20260813-santali-cluster.csv"


def _convert(value: str) -> str:
    profile = Tokenizer("conversion/santali-cluster.txt")
    return unicodedata.normalize(
        "NFC",
        profile(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_santali_cluster_clean_pdf_targets_are_complete():
    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert len(rows) == 4882
    assert all(len(row) == 11 and row[2] == row[5] for row in rows)
    assert Counter(row[0] for row in rows) == {
        "santali_rajarampur": 304,
        "santali_rautnagar": 303,
        "mundari_nijpara": 307,
        "santali_paharpur": 311,
        "mahali_abirpara": 300,
        "mahali_matindor": 303,
        "santali_patichora": 311,
        "santali_jabri": 313,
        "mundari_begunbari": 303,
        "mahali_pachondor": 305,
        "koda_kundang": 303,
        "kol_babudaing": 303,
        "koda_krishnupur": 308,
        "santali_bodobelghoria": 309,
        "mundari_karimpur": 299,
        "santali_rashidpur": 300,
    }
    assert {int(row[10].split(":")[1]) for row in rows} == (
        set(range(1, 308))
        - {124, 149, 151, 152, 194, 221, 240, 255, 257, 301, 303, 306}
    )
    assert len({row[10] for row in rows}) == 4882
    assert not ({"0", "E", "M"} & {row[0] for row in rows})
    assert all(not any(0xE000 <= ord(char) <= 0xF8FF for char in row[2]) for row in rows)


def test_santali_cluster_ipa_profile_covers_every_source_form():
    assert _convert("ʃiŋgi") == "śiŋgi"
    assert _convert("ʈʃando") == "ṭśando"
    assert _convert("rɔŋdʰɔnu") == "roŋdʰonu"
    assert _convert("pahaɽ") == "pahaṛ"
    assert _convert("d̪ɑʔɑʔ") == "dāʔāʔ"
    assert _convert("bɨndɾi") == "bɨndri"

    with open(SOURCE_FILE, encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            assert "�" not in _convert(row[2])


def test_santali_cluster_lects_are_registered_by_language():
    with open("cldf/dialects.csv", encoding="utf-8", newline="") as stream:
        dialects = {row["Source_Language_ID"]: row for row in csv.DictReader(stream)}

    expected = {
        "santali_rajarampur": ("sa", "sant1410"),
        "santali_rautnagar": ("sa", "sant1410"),
        "santali_paharpur": ("sa", "sant1410"),
        "santali_patichora": ("sa", "sant1410"),
        "santali_jabri": ("sa", "sant1410"),
        "santali_bodobelghoria": ("sa", "sant1410"),
        "santali_rashidpur": ("sa", "sant1410"),
        "mundari_nijpara": ("mu", "mund1320"),
        "mundari_begunbari": ("mu", "mund1320"),
        "mundari_karimpur": ("mu", "mund1320"),
        "mahali_abirpara": ("Mahali", "maha1291"),
        "mahali_matindor": ("Mahali", "maha1291"),
        "mahali_pachondor": ("Mahali", "maha1291"),
        "koda_kundang": ("Koda", "koda1236"),
        "koda_krishnupur": ("Koda", "koda1236"),
        "kol_babudaing": ("KolBangladesh", "kolb1241"),
    }
    for lect, (language, glottocode) in expected.items():
        assert dialects[lect]["Language_ID"] == language
        assert dialects[lect]["Glottocode"] == glottocode


def test_santali_cluster_compiled_rows_keep_raw_source_transcription():
    with open("cldf/form-source-keys.csv", encoding="utf-8", newline="") as stream:
        source_keys = [
            row for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("santali-cluster:")
        ]
    assert len(source_keys) == 4882
    assert len({row["Source_Key"] for row in source_keys}) == 4882

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
    assert len(forms) == 4882
    assert all(row["Phonemic"] for row in forms)
    assert sum(row["Form"] != row["Phonemic"] for row in forms) > 3000
