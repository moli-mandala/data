import csv
import runpy
from pathlib import Path


MODULE = runpy.run_path(Path("data/other/forms/raw_data/drasi.py"))
INPUT = MODULE["INPUT"]
parse_lines = MODULE["parse_lines"]
write_csv = MODULE["write_csv"]


def parsed_forms():
    return parse_lines(INPUT.read_text(encoding="utf-8").splitlines())


def test_drasi_parser_recovers_complete_source_vocabulary():
    forms = parsed_forms()

    # The printed source has no items 741 or 1958. All other numbered entries,
    # including one line dropped at each PDF page boundary, are represented.
    assert len({form.item for form in forms}) == 2542
    assert len(forms) == 3551
    assert not any(
        0xE000 <= ord(character) <= 0xF8FF
        for form in forms
        for character in form.form
    )
    assert all(form.form and "[" not in form.form and "]" not in form.form for form in forms)


def test_drasi_parser_repairs_known_pdf_extraction_failures():
    forms = parsed_forms()
    by_item = {}
    for form in forms:
        by_item.setdefault(form.item, []).append(form.form)

    assert by_item[36] == ["bír", "bíre"]
    assert by_item[116] == ["sõ:ʈʂi"]  # page-boundary row absent from raw text
    assert by_item[1218] == ["du:"]  # whole row absent from raw text
    assert by_item[1921] == ["ʂã: thyó:no"]
    assert by_item[2540] == ["muɣúr"]  # OCR read the source number as 2240


def test_drasi_csv_has_ingestion_schema(tmp_path):
    destination = tmp_path / "drasi.csv"
    write_csv(parsed_forms(), destination)

    with destination.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 3551
    assert {len(row) for row in rows} == {8}
    assert {row[0] for row in rows} == {"dr"}
    assert {row[7] for row in rows} == {"rajapurohit2012"}
    assert all(row[2] == row[5] for row in rows)
