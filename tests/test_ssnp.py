import csv
import importlib.util
import io
import re
import sys
import unicodedata
from pathlib import Path

from segments import Tokenizer

from make_cldf import parse_file


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/ssnp.py"
SPEC = importlib.util.spec_from_file_location("ssnp_extractor", SCRIPT)
ssnp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ssnp
SPEC.loader.exec_module(ssnp)


def transcribe_ssnp(form: str) -> str:
    profile = Tokenizer("conversion/ssnp.txt")
    return unicodedata.normalize(
        "NFC",
        profile(unicodedata.normalize("NFD", form), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_ssnp_ipa_profile_uses_house_transcription():
    assert transcribe_ssnp("wʌ'jud") == "vayūd"
    assert transcribe_ssnp("ɖhɛr") == "ḍʰer"
    assert transcribe_ssnp("t͡ɕɛ̣̃ʌ") == "ʦ̣ẹ̃a"
    assert transcribe_ssnp("ʂɪʂ") == "ṣiṣ"
    assert transcribe_ssnp("khɑ́ːɳɖo") == "kʰā́ṇḍo"


def test_ssnp_profile_covers_every_extracted_form():
    source_file = Path(__file__).parents[1] / "data/other/forms/20260725-ssnp.csv"
    with source_file.open(encoding="utf-8") as handle:
        converted = [transcribe_ssnp(row[2]) for row in csv.reader(handle)]
    assert converted
    assert not any("�" in form for form in converted)


def test_ssnp_import_keeps_ipa_and_converts_display_form():
    source_file = Path(__file__).parents[1] / "data/other/forms/20260725-ssnp.csv"
    rows, stats = parse_file(str(source_file), io.StringIO(), name="ssnp")
    assert stats == {"converted": len(rows), "for_conversion": len(rows)}
    assert rows[0].form == transcribe_ssnp(rows[0].ipa)
    assert rows[0].ipa == rows[0].old_form


def test_header_and_slash_aware_cell_parsing():
    assert ssnp.header_items(" 1. body 2. head 3. hair") == [
        ("1", "body"), ("2", "head"), ("3", "hair")
    ]
    cells, status = ssnp.split_cells("qàlip kVpàl / sor phUr", 3)
    assert cells == ["qàlip", "kVpàl / sor", "phUr"]
    assert status == "exact"


def test_wrapped_glyph_fragments_are_rejoined_at_line_boundaries():
    assert ssnp.join_fragments(["kàlip kVpàl ph", "Ur"]) == "kàlip kVpàl phUr"
    assert ssnp.join_fragments(["bux'th", "iàn"]) == "bux'thiàn"


def test_legacy_pdf_font_is_decoded_to_unicode_ipa():
    assert ssnp.decode_legacy("wV'jud S7à bàl") == "wʌ'jud ʂɑ bɑl"
    assert ssnp.decode_legacy("kyQn") == "kyæn"
    assert ssnp.decode_legacy("àKgu5{i") == "ɑŋgúʒi"
    assert ssnp.decode_legacy("nà3") == "nɑ̃"
    assert ssnp.decode_legacy("gVrà 3") == "gʌrɑ̃"
    assert ssnp.decode_legacy("r†àt") == "ɽɑt"


def test_all_sources_extract_numbered_forms():
    rows = ssnp.extract()
    sources = {row.source_file for row in rows}
    assert sources == {"chitral", "hindko", "kohistani", "gojri", "indus kohistani", "ushojo"}
    assert len(rows) > 10_000
    assert any(row.location == "JAM" and row.number == "1" and row.form == "pInDà / jIsV" for row in rows)
    assert any(row.location == "USH" and row.number == "1" and row.form == "'wàjuth" for row in rows)
    assert any(row.number.startswith("supp-") and row.gloss == "funeral bier" for row in rows)
    assert not any("Word Lists" in row.form for row in rows)


def test_each_volume_has_its_own_bibliographic_source():
    assert ssnp.SOURCE_BY_FILE == {
        "chitral": "decker1992",
        "hindko": "rensch-hallberg-oleary1992",
        "gojri": "rensch-hallberg-oleary1992",
        "kohistani": "rensch-decker-hallberg1992",
        "indus kohistani": "rensch-decker-hallberg1992",
        "ushojo": "rensch-decker-hallberg1992",
    }


def test_every_survey_variety_has_language_metadata():
    root = Path(__file__).parents[1]
    with (root / "cldf/languages.csv").open(encoding="utf-8") as handle:
        language_ids = {row["ID"] for row in csv.DictReader(handle)}
    with (root / "cldf/dialects.csv").open(encoding="utf-8") as handle:
        source_ids = {row["Source_Language_ID"] for row in csv.DictReader(handle)}
    extracted_ids = {ssnp.language_id(row) for row in ssnp.extract()}
    assert extracted_ids <= language_ids | source_ids


def test_batera_is_canonical_bhateri():
    batera = ssnp.Entry("indus kohistani", "BAT", "1", "body", "", "exact", "")
    assert ssnp.language_id(batera) == "bhatr"

    language_file = Path(__file__).parents[1] / "cldf/languages.csv"
    with language_file.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Name"] for row in rows if row["Glottocode"] == "bate1261"] == ["Bhateri"]


def test_all_import_rows_survive_cldf_ingestion():
    source_file = Path(__file__).parents[1] / "data/other/forms/20260725-ssnp.csv"
    cldf_file = Path(__file__).parents[1] / "cldf/forms.csv"
    with source_file.open(encoding="utf-8") as handle:
        source_rows = list(csv.reader(handle))
    assert not any(re.search(r"\d|Word Lists|Survey Data|Appendix", row[2]) for row in source_rows)
    assert not any(any(char in row[2] for char in "Œ†ƒ½{}VFKQ") for row in source_rows)
    with cldf_file.open(encoding="utf-8") as handle:
        ingested_rows = [
            row for row in csv.DictReader(handle)
            if any(source in row["Source"] for source in {
                "decker1992", "rensch-decker-hallberg1992", "rensch-hallberg-oleary1992",
            })
        ]
    # A small number coalesce with an already-present identical form during unification.
    assert len(ingested_rows) >= 0.99 * len(source_rows)
    with (Path(__file__).parents[1] / "cldf/dialects.csv").open(encoding="utf-8") as handle:
        canonical = {
            row["Source_Language_ID"]: row["Language_ID"]
            for row in csv.DictReader(handle) if row["Source_Language_ID"]
        }
    ingested = {
        (row["Language_ID"], row["Phonemic"], row["Gloss"])
        for row in ingested_rows
    }
    source = {
        (canonical.get(row[0], row[0]), row[5], row[3])
        for row in source_rows
    }
    assert len(ingested & source) >= 0.99 * len(source)
    assert all("[" in row["Source"] for row in ingested_rows)


def test_every_survey_location_has_coordinates():
    dialect_file = Path(__file__).parents[1] / "cldf/dialects.csv"
    with dialect_file.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["Source_Language_ID"].startswith("SSNP-")
        ]
    assert rows
    assert all(row["Latitude"] and row["Longitude"] for row in rows)
