import importlib.util
import csv
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/sigiri.py"
SPEC = importlib.util.spec_from_file_location("sigiri_extractor", SCRIPT)
sigiri = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sigiri
SPEC.loader.exec_module(sigiri)


def test_normalize_headword_repairs_macron_and_velar_nasal():
    assert sigiri.normalize_headword("äñg") == "āṅg"
    assert sigiri.normalize_headword("ańñga!") == "aṅga"


def test_extract_gloss_after_etymology_and_before_citations():
    entry = "ak, s. [Skt. agra, P. agga; cf. Dhpr. 69], end. See nātak."
    assert sigiri.extract_gloss(entry, "ak") == "end"

    entry = "a, prt. [Skt. āgata], come, arrived, 48, 59, 70."
    assert sigiri.extract_gloss(entry, "a") == "come, arrived"


def test_printed_glossary_descriptors_become_canonical_tags():
    assert sigiri.grammatical_tags("ak, s. [Skt. agra], end") == ["noun"]
    assert sigiri.grammatical_tags("aṅgana, s.f. [Skt. aṅganā]") == ["noun", "f"]
    assert sigiri.grammatical_tags("a, prt. [Skt. āgata]") == ["part"]
    assert sigiri.grammatical_tags("x, inst. of y") == ["instr"]


def test_extracts_sanskrit_etyma_and_matches_only_unique_cdial_heads():
    assert sigiri.extract_sanskrit_etyma(
        "agu, s. [Skt. agra+ka], border, 366"
    ) == ["agra+ka"]
    assert sigiri.extract_sanskrit_etyma(
        "a, prt. [Skt. āgata, e.f. ā°], come"
    ) == ["āgata"]

    index = {
        sigiri.normalize_sanskrit_etymon("āgata"): {"1045"},
        sigiri.normalize_sanskrit_etymon("ṭaṅka"): {"5426", "5427"},
    }
    assert sigiri.match_cdial_ids(["agata"], index) == (["1045"], False)
    assert sigiri.match_cdial_ids(["taṅka"], index) == ([], True)

    cdial = sigiri.load_cdial_headword_index()
    assert sigiri.match_cdial_ids(["āgata"], cdial) == (["1045"], False)
    assert sigiri.match_cdial_ids(["kasta"], cdial) == ([], True)


def test_parse_pages_keeps_column_and_page_provenance():
    pages = [
        {
            "pdf_page": 450,
            "columns": [
                {
                    "column": 1,
                    "lines": [
                        {"text": "ak, s. [Skt. agra], end.", "left": 82, "top": 20, "confidence": 92},
                        {"text": "See nātak.", "left": 145, "top": 40, "confidence": 90},
                        {"text": "aga, s. [Skt. aṅga], limb, 379.", "left": 86, "top": 60, "confidence": 88},
                    ],
                }
            ],
        }
    ]
    entries = sigiri.parse_pages(pages)

    assert [entry.headword for entry in entries] == ["ak", "aga"]
    assert [entry.gloss for entry in entries] == ["end", "limb"]
    assert entries[0].pdf_page == 450
    assert entries[0].printed_page == 442
    assert entries[0].column == 1


def test_entry_start_uses_hanging_indent_and_grammar_label():
    assert sigiri.is_entry_start(sigiri.OCRLine("a, indec. [Skt. ā]", 80, 10, 90))
    assert sigiri.is_entry_start(sigiri.OCRLine("aṅgana, s.f. [Skt. aṅganā]", 19, 10, 90))
    assert not sigiri.is_entry_start(sigiri.OCRLine("limb, 379.", 145, 20, 90))
    assert not sigiri.is_entry_start(sigiri.OCRLine("See aṅg-rā,", 150, 30, 90))
    assert sigiri.is_entry_start(sigiri.OCRLine("anurā, s. [Skt. anurāga]", 163, 40, 90))
    assert not sigiri.is_entry_start(sigiri.OCRLine("v. 255, var. agan", 54, 50, 90))


def test_detect_column_crops_uses_vertical_rules():
    from PIL import Image, ImageDraw

    image = Image.new("L", (1000, 900), 245)
    draw = ImageDraw.Draw(image)
    draw.line((350, 312, 350, 899), fill=20, width=2)
    draw.line((650, 312, 650, 899), fill=20, width=2)
    crops, rules = sigiri.detect_column_crops(image)

    assert rules == (350, 650)
    assert crops[0][2] < crops[1][0] + 50
    assert crops[1][2] < crops[2][0] + 50


def test_rule_plausibility_accounts_for_recto_verso_registration():
    assert sigiri.plausible_rules((790, 1350), 2288, 431)
    assert sigiri.plausible_rules((880, 1480), 2288, 432)
    assert not sigiri.plausible_rules((629, 1450), 2288, 436)


def test_generated_sigiri_import_has_complete_ingestion_schema():
    source = Path(__file__).parents[1] / "data/other/forms/20260726-paranavitana-sigiri.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    # Five entries cite two independently matchable Sanskrit alternatives.
    assert len(rows) == 2874
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"OSi"}
    assert all(row[7].startswith("paranavitana[p. ") for row in rows)
    assert all(not row[6] for row in rows)
    assert all(row[2] for row in rows)
    assert any(row[1] for row in rows)
    assert any("[p. 431 " in row[7] for row in rows)
    assert any("[p. 480 " in row[7] for row in rows)
    assert any("noun" in row[14].split() for row in rows)
