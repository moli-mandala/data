import csv
from pathlib import Path
import sys


RAW_DATA = Path(__file__).parents[1] / "data" / "other" / "forms" / "raw_data"
sys.path.insert(0, str(RAW_DATA))

from ghatage_survey import VOLUMES, parse_page, rich_row  # noqa: E402


def _tsv(*rows):
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    return "\n".join([header, *rows])


def _word(number, left, top, width, confidence, text):
    return f"5\t1\t1\t1\t1\t{number}\t{left}\t{top}\t{width}\t30\t{confidence}\t{text}"


def test_parse_page_uses_columns_pos_and_confidence():
    volume = VOLUMES["marati-kasargod"]
    entries = parse_page(
        _tsv(
            _word(1, 220, 253, 72, 96, "addi"),
            _word(2, 315, 253, 79, 96, "Adv."),
            _word(3, 741, 253, 113, 81, "before."),
            _word(4, 220, 300, 110, 45, "andaji"),
            _word(5, 356, 300, 36, 90, "N."),
            _word(6, 742, 300, 77, 96, "idea,"),
            _word(7, 846, 300, 139, 96, "thought."),
        ),
        volume,
        144,
    )
    assert [(entry.form, entry.pos, entry.definition) for entry in entries] == [
        ("addi", "Adv.", "before."),
        ("andaji", "N.", "idea, thought."),
    ]
    assert entries[1].flags == ["low-confidence"]
    assert entries[1].key(volume.source_id) == "ghatage-kasargod1970:p136:e2"


def test_rich_row_has_stable_source_and_review_tag():
    volume = VOLUMES["marati-kasargod"]
    entry = parse_page(
        _tsv(
            _word(1, 220, 253, 72, 45, "addi"),
            _word(2, 315, 253, 79, 96, "Adv."),
            _word(3, 741, 253, 113, 81, "before."),
        ),
        volume,
        145,
    )[0]
    row = rich_row(entry, volume)
    assert len(row) == 15
    assert row[7] == "ghatage-kasargod1970[p. 137, entry 1]"
    assert row[10] == "ghatage-kasargod1970:p137:e1"
    assert row[5] == ""
    assert row[14] == (
        "adv dialect:M:ghatage-kasargod1970:Marati%20of%20Kasargod ocr-review"
    )


def test_bare_quote_does_not_join_following_tsv_lines():
    volume = VOLUMES["marati-kasargod"]
    entries = parse_page(
        _tsv(
            _word(1, 220, 253, 72, 96, "gini."),
            _word(2, 741, 253, 113, 96, "quickly."),
            _word(3, 1300, 253, 5, 0, '"'),
            _word(4, 220, 300, 110, 96, "andaji"),
            _word(5, 356, 300, 36, 90, "N."),
            _word(6, 742, 300, 77, 96, "idea."),
        ),
        volume,
        144,
    )
    assert [(entry.form, entry.pos, entry.definition) for entry in entries] == [
        ("gini.", "", "quickly."),
        ("andaji", "N.", "idea."),
    ]


def test_ocr_punctuation_variant_is_canonicalized_as_pos():
    volume = VOLUMES["marati-kasargod"]
    entry = parse_page(
        _tsv(
            _word(1, 220, 253, 150, 80, "hogalpt"),
            _word(2, 390, 253, 40, 70, "V_"),
            _word(3, 741, 253, 100, 90, "abuse."),
        ),
        volume,
        176,
    )[0]
    assert (entry.form, entry.pos) == ("hogalpt", "V.")


def test_printer_footer_is_not_appended_to_last_definition():
    volume = VOLUMES["marati-kasargod"]
    entries = parse_page(
        _tsv(
            _word(1, 220, 1900, 100, 90, "ho:lt"),
            _word(2, 350, 1900, 40, 90, "N."),
            _word(3, 741, 1900, 70, 90, "hall."),
            _word(4, 741, 1950, 90, 90, "PRESS,"),
            _word(5, 850, 1950, 90, 90, "BOMBAY"),
            _word(6, 741, 1980, 20, 90, "4"),
        ),
        volume,
        176,
    )
    assert entries[0].definition == "hall."


def test_composite_and_noisy_pos_do_not_leak_into_form():
    volume = VOLUMES["marati-kasargod"]
    entries = parse_page(
        _tsv(
            _word(1, 220, 300, 100, 90, "misyo"),
            _word(2, 340, 300, 100, 80, "M.F,"),
            _word(3, 460, 300, 50, 80, "Pl."),
            _word(4, 741, 300, 100, 90, "people."),
            _word(5, 220, 350, 100, 90, "vori,"),
            _word(6, 741, 350, 100, 90, "boon."),
        ),
        volume,
        161,
    )
    assert (entries[0].form, entries[0].pos) == ("misyo", "M.F.")
    assert (entries[1].form, entries[1].pos) == ("vori", "")


def test_checked_in_audit_and_seeded_sample_account_for_ocr_review():
    forms_path = RAW_DATA.parent / "20260817-ghatage-marati-kasargod.csv"
    audit_path = RAW_DATA / "20260817-ghatage-marati-kasargod-audit.csv"
    sample_path = RAW_DATA / "20260817-ghatage-marati-kasargod-sample.csv"

    with forms_path.open(encoding="utf-8", newline="") as handle:
        installed = list(csv.reader(handle))
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    with sample_path.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))

    emitted = sum(len(row["Emitted_Keys"].split("|")) for row in audit if row["Emitted_Keys"])
    assert len(installed) == emitted == 1271
    assert len(audit) == 1244
    assert {row["Status"] for row in audit} == {"verified", "ocr_unreviewed"}
    assert all("ocr-review" in row[14].split() for row in installed)
    assert len(sample) == 20
    assert {row["Seed"] for row in sample} == {"1970"}
    assert {row["Final_Result"] for row in sample} == {"PASS"}


def test_source_image_corrections_have_unique_stable_keys():
    path = RAW_DATA / "20260817-ghatage-marati-kasargod-corrections.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["Entry_Key"] for row in rows]
    assert len(keys) == len(set(keys)) == 129
    assert {
        "ghatage-kasargod1970:p151:manual:doggadogga",
        "ghatage-kasargod1970:p155:manual:pu-ra",
        "ghatage-kasargod1970:p159:manual:madi",
    } <= set(keys)
