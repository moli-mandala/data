import csv
from collections import defaultdict
from pathlib import Path


CDIAL = Path(__file__).parents[1] / "data" / "cdial" / "cdial.csv"
CORRUPT = Path(__file__).parents[1] / "data" / "cdial" / "corrupt_forms.csv"


def parsed_entries():
    entries = defaultdict(list)
    with CDIAL.open(newline="") as handle:
        for row in csv.reader(handle):
            entries[row[1]].append(row)
    return entries


def forms(rows, language):
    return [row[2] for row in rows if row[0] == language]


def test_undecodable_source_forms_are_audited_instead_of_installed():
    with CORRUPT.open(encoding="utf-8", newline="") as handle:
        corrupt = list(csv.DictReader(handle))

    assert {(row["Entry_ID"], row["Language_ID"]) for row in corrupt} == {
        ("1979", "Ash"),
        ("6498", "K"),
    }
    assert all(row["Status"] == "excluded" for row in corrupt)
    assert all(
        any(0x7F <= ord(character) < 0xA0 for character in row["Raw_Form"])
        for row in corrupt
    )

    for rows in parsed_entries().values():
        assert all(
            not any(ord(character) < 0x20 or 0x7F <= ord(character) < 0xA0 for character in row[2])
            for row in rows
        )


def test_dialect_qualifiers_do_not_duplicate_parent_language_rows():
    entries = parsed_entries()
    assert forms(entries["10754"], "eur") == ["ruv"]
    assert forms(entries["10754"], "Gy") == []
    assert forms(entries["8388"], "awāṇ") == ["pētlā"]
    assert forms(entries["8388"], "L") == []


def test_consecutive_one_letter_languages_are_not_mistaken_for_author_initials():
    rows = parsed_entries()["14226"]
    assert all(forms(rows, language) == ["bača"] for language in ("S", "L", "P", "WPah"))


def test_explanatory_forms_do_not_become_reflex_rows():
    entries = parsed_entries()
    assert forms(entries["51"], "Kho") == ["aii", "hai"]
    assert forms(entries["6366"], "Pk") == ["ḍīṇa", "ḍippaï"]
    assert forms(entries["9420"], "Md") == ["vī"]


def test_language_matching_does_not_start_inside_prose_or_after_loan_arrows():
    entries = parsed_entries()
    assert forms(entries["3904"], "M") == ["khubā", "khubaṛ"]
    assert forms(entries["487"], "Pa") == ["apidahati"]
    assert forms(entries["487"], "Si") == ["vahanavā"]
    assert forms(entries["571"], "Pa") == ["amata"]
    assert forms(entries["571"], "Pk") == ["amaya", "amiya", "amuya"]


def test_parenthetical_relations_do_not_suppress_later_real_forms():
    rows = parsed_entries()["5839"]
    assert forms(rows, "Paš") == ["tīṣ", "tēṣarī́"]
    assert forms(rows, "ar") == ["tinigó"]


def test_parenthetical_letters_apostrophes_and_degree_abbreviations_are_preserved():
    entries = parsed_entries()
    assert "γeč(h)" in forms(entries["43"], "Kho")
    assert "g'ā̃s" in forms(entries["4471"], "Gaw")
    assert "karāhā" in forms(entries["2638"], "P")
    assert "karāi" in forms(entries["2638"], "Or")


def test_page_layout_markup_does_not_leak_into_notes():
    for row in parsed_entries()["11559"]:
        assert "<div>" not in row[6]
        assert "<hw>" not in row[6]


def test_comma_linked_forms_and_languages_share_the_printed_gloss():
    rows = parsed_entries()["2854"]
    by_form = {(row[0], row[2], row[8]): row[3] for row in rows}

    # A quote after the second Pali form scopes backward over both forms.
    assert by_form[("Pk", "kattaï", "2:")] == "cuts"
    assert by_form[("Pk", "kaṭṭaï", "2:")] == "cuts"
    # A quote after Maithili scopes forward across comma-linked language labels.
    assert by_form[("Mth", "kāṭab", "2:")] == "to cut"
    assert by_form[("lakh", "kāṭab", "2:")] == "to cut"
    assert by_form[("H", "kāṭnā", "2:")] == "to cut"
    assert by_form[("G", "kāṭvũ", "2:")] == "to cut"
    # The same applies within the explicitly marked intransitive run.
    assert by_form[("H", "kaṭnā", "4:Intr. with <i>a</i>")] == "to be cut"
    # A causative label begins a new run; do not copy "to cut, fell" onto it.
    assert by_form[("kṭg", "kəṭauṇõ", "5:")] == ""


def test_intervening_morphology_stops_same_language_gloss_propagation():
    rows = parsed_entries()["841"]
    pali = {row[2]: row[3] for row in rows if row[0] == "Pa"}

    assert pali["ōvahati"] == ""
    assert pali["ōvuyhati"] == "is carried down (a river)"


def test_numbered_and_causative_head_forms_do_not_inherit_base_glosses():
    entries = parsed_entries()

    entry_2058 = {row[2]: row[3] for row in entries["2058"] if row[0] == "Indo-Aryan"}
    assert entry_2058["*udraṁhati"] == "jumps up"
    assert entry_2058["*udraṁhayati"] == ""

    entry_9139 = {row[2]: row[3] for row in entries["9139"] if row[0] == "Indo-Aryan"}
    assert entry_9139["bandhati"] == "binds"
    assert entry_9139["bandháyati"] == ""


def test_mixed_definition_spans_do_not_carry_one_meaning_to_the_next_language():
    entries = parsed_entries()

    # Pa. has both "measures" and "measurement" immediately before these mixed verb/noun forms.
    prakrit_10132 = {row[2]: row[3] for row in entries["10132"] if row[0] == "Pk"}
    assert prakrit_10132["miṇaï"] == ""
    assert prakrit_10132["miṇaṇa"] == ""

    # OAw. has citerā "painter" and citeraï "paints"; lakh. citērā cannot safely inherit either.
    lakh_4805 = [row for row in entries["4805"] if row[0] == "lakh"]
    assert len(lakh_4805) == 1
    assert lakh_4805[0][3] == ""


def test_unglossed_direct_reflexes_inherit_the_headword_definition():
    entries = parsed_entries()

    mother = {(row[0], row[2]): row[3] for row in entries["5100"]}
    assert mother[("Pa", "jananī")] == "mother"
    assert mother[("Pk", "jaṇaṇī")] == "mother"
    assert mother[("P", "jaṇṇī")] == "mother"

    shoe = {(row[0], row[2]): row[3] for row in entries["3127"]}
    assert shoe[("Pa", "kaṭṭhapādukā")] == "wooden shoe"
    assert shoe[("Pk", "kaṭṭhapāuyā")] == "wooden shoe"
    assert shoe[("Or", "kaṭhaü")] == "wooden shoe"
    # The following "Other NIA forms" subgroup is not a direct-reflex run.
    assert shoe[("K", "khrāv")] == ""


def test_nearest_following_definition_scopes_over_a_local_form_run():
    entries = parsed_entries()

    bones = {(row[0], row[2]): row[3] for row in entries["13952"]}
    assert bones[("L", "haḍḍ")] == "bone"
    assert bones[("L", "haḍḍī")] == "bone"
    assert bones[("L", "haḍḍā")] == "spavin"
    assert bones[("P", "haḍḍ")] == "bone"

    young = {(row[0], row[2]): row[3] for row in entries["5712"]}
    assert young[("Pk", "taruṇa")] == "young, fresh"
    assert young[("Pk", "taruṇaya")] == "young, fresh"
    assert young[("Pk", "taluṇa")] == "young, fresh"
    assert young[("Pk", "taruṇī")] == "young woman"
