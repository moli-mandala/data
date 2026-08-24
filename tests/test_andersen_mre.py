import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/andersen_mre.py"
SPEC = importlib.util.spec_from_file_location("andersen_mre_extractor", SCRIPT)
andersen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = andersen
SPEC.loader.exec_module(andersen)


def test_sloping_margin_keeps_heads_and_rejects_indented_quoted_prose():
    page = {
        "pdf_page": 137,
        "width": 1425,
        "lines": [
            {"text": 'aya- a. "worthy"', "left": 89, "top": 200, "confidence": 95},
            {"text": 'in Pali = rāja or "rulers"', "left": 147, "top": 400, "confidence": 90},
            {"text": 'avaradhiya- m. "minimum"', "left": 83, "top": 900, "confidence": 94},
            {"text": 'as- "to exist"', "left": 77, "top": 1500, "confidence": 92},
        ],
    }
    keys = andersen.entry_start_keys(page)
    assert keys == {(200, 89), (900, 83), (1500, 77)}


def test_verified_ocr_repairs_preserve_dictionary_spelling():
    assert andersen.normalize_headword("ațha-") == "aṭha-"
    assert andersen.normalize_headword("aqdhatiya-") == "aḍhatiya-"
    assert andersen.normalize_headword("vampa-") == "vaṃṇa-"
    assert andersen.normalize_headword("Saca-") == "śaca-"


def test_printed_dictionary_descriptors_become_canonical_tags():
    assert andersen.tags_for_pos("m") == ["noun", "m"]
    assert andersen.tags_for_pos("a") == ["adj"]
    assert andersen.tags_for_pos("pres") == ["verb", "pres"]
    assert andersen.tags_for_pos("cf") == []


def test_minor_edict_findspots_become_dialect_labels():
    raw = "-e (NomSg) 1H [(Ms)], 1N [Br,(Ni),Ud,Ru,Pn,Ga,P1]"
    assert andersen.extract_dialects(raw) == [
        "Brahmagiri", "Gavimath", "Maski", "Nittur", "Panguraria",
        "Palkigundu", "Rupnath", "Udegolam",
    ]
    assert andersen.dialect_tag("Brahmagiri") == (
        "dialect:As:as-brahmagiri:Brahmagiri"
    )
    assert andersen.dialect_tag("Jatinga-Ramesvara") == (
        "dialect:As:as-jatinga-ramesvara:Jatinga-Ramesvara"
    )


def test_andersen_profile_uses_house_aspiration_and_anusvara():
    profile = Path(__file__).parents[1] / "conversion/andersen.txt"
    with profile.open(encoding="utf-8", newline="") as handle:
        mappings = {row["Grapheme"]: row["IPA"] for row in csv.DictReader(handle, delimiter="\t")}
    assert mappings["dh"] == "dʰ"
    assert mappings["ḍh"] == "ḍʰ"
    assert mappings["ṃ"] == "ṁ"


def test_generated_andersen_import_is_complete_and_auditable():
    source = Path(__file__).parents[1] / "data/other/forms/20260804-andersen-mre.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 287
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"As"}
    assert all(row[7].startswith("andersen1990[p. ") for row in rows)
    assert all(not row[6] for row in rows)
    assert all(row[2] for row in rows)
    assert any(row[2] == "saṃyata-" and row[3] == "restrained" for row in rows)
    assert any(
        "dialect:As:as-brahmagiri:Brahmagiri" in row[14].split()
        for row in rows
    )
    assert any(
        "dialect:As:as-jatinga-ramesvara:Jatinga-Ramesvara" in row[14].split()
        for row in rows
    )
    assert any("noun" in row[14].split() for row in rows)
    assert any("adj" in row[14].split() for row in rows)
    assert any("[p. 134 (printed p. 136)]" in row[7] for row in rows)
    assert any("[p. 177 (printed p. 179)]" in row[7] for row in rows)
