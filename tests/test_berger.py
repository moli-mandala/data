import importlib.util
import csv
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/berger.py"
SPEC = importlib.util.spec_from_file_location("berger_extractor", SCRIPT)
berger = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = berger
SPEC.loader.exec_module(berger)


def test_repair_trailing_question_mark_read_as_digit():
    assert berger.repair_id("114337", {"11433"}) == ("11433", "repaired")
    assert berger.repair_id("1464", {"1464"}) == ("1464", "exact")


def test_extract_simple_and_compound_headwords():
    assert berger.extract_form("abáś y hz.ng. Schwierigkeit (sh. abáś, T 11433?)") == "abáś"
    assert berger.extract_form("baláac̣ man-́ nach einem Wurf aufrecht stehen (T 6658)") == "baláac̣ man-́"


def test_yasin_variant_becomes_werchikwar_form():
    text = "áḍa verbleibend, übrig (ys. áḍe, vgl. T 644)"
    assert berger.yasin_variant(text) == "áḍe"


def test_candidate_page_number_uses_side_of_spread():
    data = {
        "pdf_page": 9,
        "width": 3500,
        "paragraphs": [
            {"text": "áḍa verbleibend (ys. áḍe, vgl. T 644)", "left": 100, "top": 1, "confidence": 90},
            {"text": "adít y Sonntag (sh. adít, T 1154)", "left": 2000, "top": 1, "confidence": 90},
        ],
    }
    rows = berger.parse_pages([data], {"644", "1154"})
    assert rows[0].printed_page == 12
    assert rows[-1].printed_page == 13


def test_turner_pattern_does_not_treat_t_followed_by_letters_as_an_id():
    assert not berger.TURNER_RE.search("T Lorimer")
    assert not berger.TURNER_RE.search("der Tatsache")


def test_full_lexicon_keeps_unlinked_rows_and_rich_metadata():
    pages = [{
        "pdf_page": 8,
        "width": 3500,
        "paragraphs": [{
            "text": "abáś y hz.ng. Schwierigkeit (aus Shina abaś)",
            "left": 158,
            "top": 100,
            "confidence": 95,
        }],
    }]
    entries = berger.parse_lexicon_pages(pages, set())
    assert entries[0].cdial_id == ""
    assert entries[0].entry_key
    assert {"noun", "dialect:Hunza", "dialect:Nager"} <= set(entries[0].tags)
    assert entries[0].etymology == "aus Shina abaś"
    assert list(berger.import_rows(entries))[0][1] == ""


def test_explicit_noun_class_is_kept_alongside_adverb_use():
    tags = berger._grammar_tags(
        "abáś y hz.ng. Schwierigkeit, Unglück; adv. schwierig", "abáś"
    )
    assert {"noun", "adv", "dialect:Hunza", "dialect:Nager"} <= set(tags)


def test_installed_gold_tranche_has_rich_rows_and_grammar_audit():
    root = Path(__file__).parents[1]
    gold_path = root / "data/other/forms/20220930-berger.csv"
    audit_path = root / "data/other/forms/raw_data/20220930-berger-grammar-audit.csv"
    if not audit_path.exists():
        return
    gold = list(csv.reader(gold_path.open(encoding="utf-8")))
    audit = list(csv.DictReader(audit_path.open(encoding="utf-8")))
    assert len(gold) == len(audit) == 39
    assert {len(row) for row in gold} == {15}
    assert any("noun" in row[14].split() for row in gold)
    assert any("verb" in row[14].split() for row in gold)
    assert {row["Strategy"] for row in audit} <= {
        "aligned-source", "exact-form-source", "fuzzy-form-source",
        "legacy-printed-evidence",
    }


def test_yasin_variant_has_language_dialect_and_parent_key():
    pages = [{
        "pdf_page": 8,
        "width": 3500,
        "paragraphs": [{
            "text": "áḍa h verbleibend, übrig (ys. áḍe, vgl. T 644)",
            "left": 158,
            "top": 100,
            "confidence": 95,
        }],
    }]
    base, variant = berger.parse_lexicon_pages(pages, {"644"})
    assert (variant.language, variant.form) == ("Werch", "áḍe")
    assert variant.variant_of_key == base.entry_key
    assert {"dialect:Yasin", "alternate"} <= set(variant.tags)
    assert variant.cdial_id == "644"


def test_separate_source_paragraph_updates_parent_and_existing_variant():
    pages = [{
        "pdf_page": 8,
        "width": 3500,
        "paragraphs": [
            {
                "text": "áḍa h verbleibend (ys. áḍe)",
                "left": 158,
                "top": 100,
                "confidence": 95,
            },
            {
                "text": "(vgl. T 644)",
                "left": 158,
                "top": 130,
                "confidence": 95,
            },
        ],
    }]
    base, variant = berger.parse_lexicon_pages(pages, {"644"})
    assert base.cdial_id == variant.cdial_id == "644"
    assert "vgl. T 644" in base.etymology
    assert variant.etymology == base.etymology


def test_import_drops_relation_to_parent_supplied_by_legacy_gold_row():
    pages = [{
        "pdf_page": 8,
        "width": 3500,
        "paragraphs": [{
            "text": "áḍa h verbleibend (ys. áḍe, vgl. T 644)",
            "left": 158,
            "top": 100,
            "confidence": 95,
        }],
    }]
    base, variant = berger.parse_lexicon_pages(pages, {"644"})
    base.gold_row = 1
    rows = list(berger.import_rows([base, variant]))
    assert len(rows) == 1
    assert rows[0][2] == "áḍe"
    assert rows[0][11] == ""


def test_scan_verified_ruus_entry_boundary_repair():
    entries = [
        berger.Entry("Bur", 187, 369, "rúu-rúu ét-", "(Flugzeug)", "", "", "", "", 90, "unlinked", entry_key="berger-entry-7806"),
        berger.Entry("Bur", 187, 369, "brummen", "dröhnen. rúus, ng. rúuś Vergeltung, Rache, Heimzahlen", "10856", "", "", "", 90, "exact", entry_key="berger-entry-7807"),
        berger.Entry("Bur", 187, 369, "rúuś", "dröhnen. rúus, ng. rúuś Vergeltung, Rache, Heimzahlen", "10856", "", "", "", 90, "exact", entry_key="berger-entry-7807-dialect-1"),
    ]

    berger.apply_reviewed_repairs(entries)

    assert (entries[0].form, entries[0].gloss) == (
        "rúu-rúu ét-", "(Flugzeug) brummen, dröhnen"
    )
    assert [(entry.form, entry.gloss) for entry in entries[1:]] == [
        ("rúus", "Vergeltung, Rache, Heimzahlen"),
        ("rúuś", "Vergeltung, Rache, Heimzahlen"),
    ]
    assert {entry.printed_page for entry in entries} == {367}
