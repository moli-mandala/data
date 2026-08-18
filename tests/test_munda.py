import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from make_cldf import format_munda_parameter


ROOT = Path(__file__).parents[1]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_munda_parameter_etymology_maps_to_the_dedicated_field():
    source = [
        "m1",
        "*daˀk",
        "",
        "water",
        "Pinnow: V2.<br>MKCD: *\\*diʔaak > \\*ɗaak* 'water' (274).",
    ]

    assert format_munda_parameter(source) == [
        "m1",
        "*daˀk",
        "PMu",
        "water",
        "",
        source[4],
    ]


def test_all_munda_headwords_preserve_free_text_etymology_in_compiled_cldf():
    with (ROOT / "data/munda/params.csv").open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.reader(handle))

    aliases = {
        row["Legacy_ID"]: row["Form_ID"]
        for row in read_rows(ROOT / "cldf/form-id-aliases.csv")
    }
    forms = {row["ID"]: row for row in read_rows(ROOT / "cldf/forms.csv")}
    texts = {
        aliases.get(row["Form_ID"], row["Form_ID"]): row
        for row in read_rows(ROOT / "cldf/entry-texts.csv")
        if row["Source"] == "rau"
    }

    assert len(source_rows) == 127
    assert len(texts) == 127
    for source in source_rows:
        entry_id, headword, _language, gloss, etymology = source
        form_id = aliases.get(entry_id, entry_id)
        entry = forms[form_id]
        block = texts[form_id]

        assert entry["Language_ID"] == "PMu"
        assert entry["Form"] == headword.split(",")[0].strip()
        assert entry["Gloss"] == gloss
        assert entry["Description"] == ""
        assert entry["Etymology"] == etymology
        assert block == {
            "Form_ID": form_id,
            "Position": "0",
            "Kind": "etymology",
            "Format": "markdown",
            "Content": etymology,
            "Source": "rau",
        }
