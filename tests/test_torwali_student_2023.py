import csv
import io
import json
from collections import Counter
from pathlib import Path

from make_cldf import parse_file


ROOT = Path(__file__).parents[1]
FORMS = ROOT / "data/other/forms/20260820-torwali-student-2023.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260820-torwali-student-2023-audit.csv"
SAMPLE = ROOT / "data/other/forms/raw_data/20260820-torwali-student-2023-sample.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260820-torwali-student-2023-manifest.json"


def read_forms():
    with FORMS.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_snapshot_reconciles_every_pdf_headword_anchor():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2_269
    assert len({row["Entry_Key"] for row in rows}) == 2_269
    assert Counter(row["Status"] for row in rows) == {
        "installed": 1_943,
        "excluded_no_ipa": 326,
    }
    assert all(row["Record_SHA256"] for row in rows)
    assert all(row["Source"].startswith("torwali2023student[p. ") for row in rows)
    assert not any("unrecognized POS" in row["Reason"] for row in rows)


def test_installed_rows_preserve_source_ipa_and_do_not_publish_bad_unicode_headwords():
    rows = read_forms()
    assert len(rows) == 1_943
    assert all(len(row) == 15 for row in rows)
    assert all(row[0] == "Tor" and row[2] and row[5] == row[2] for row in rows)
    assert all(not row[4] for row in rows)  # unreliable PDF ToUnicode map
    assert len({row[10] for row in rows}) == len(rows)
    assert all(row[7].startswith("torwali2023student[p. ") for row in rows)
    assert all("�" not in field for row in rows for field in row)


def test_homographs_and_source_blank_glosses_are_not_collapsed_or_invented():
    rows = read_forms()
    by_native = {}
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    for row in audit:
        by_native.setdefault(row["Headword"], []).append(row)
    assert len(by_native["آبادی"]) == 3
    assert len({row["Entry_Key"] for row in by_native["آبادی"]}) == 3
    blank = [row for row in rows if not row[3]]
    assert {row[10] for row in blank} == {
        "torwali2023student:p019:cR:e03",
        "torwali2023student:p111:cR:e03",
        "torwali2023student:p149:cL:e01",
    }


def test_seeded_page_image_review_has_zero_material_errors():
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert {row["Material_Error"] for row in rows} == {"no"}
    assert all(row["Review"].startswith("source-image-verified") for row in rows)
    assert {row["Status"] for row in rows} == {"installed", "excluded_no_ipa"}


def test_manifest_records_identity_license_scope_and_exclusions():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf"]["sha256"] == (
        "da338088c2674f0eccdb426fbcac210ebd9d258b2cfa7f2870171b228f42e33f"
    )
    assert manifest["pdf"]["pages"] == 232
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["webonary"]["reported_entries"] == 2_271
    assert manifest["extraction"]["status_counts"] == {
        "installed": 1_943,
        "excluded_no_ipa": 326,
    }
    assert manifest["extraction"]["seeded_sample_material_errors"] == 0
    assert manifest["extraction"]["dialect_counts"] == {
        "Sinkaen/Bahrain": 2_225,
        "Chail": 44,
    }
    assert manifest["extraction"]["installed_dialect_counts"] == {
        "Sinkaen/Bahrain": 1_922,
        "Chail": 21,
    }
    assert "ToUnicode" in manifest["scope"]["excluded"]


def test_source_profile_converts_display_form_and_retains_exact_phonemic_ipa():
    rows, stats = parse_file(str(FORMS), io.StringIO())
    assert len(rows) == 1_943
    assert stats == {"converted": 1_943, "for_conversion": 1_943}
    by_key = {row.entry_key: row for row in rows}
    assert (by_key["torwali2023student:p013:cR:e02"].form,
            by_key["torwali2023student:p013:cR:e02"].ipa) == ("aūzār", "au:za:r")
    assert (by_key["torwali2023student:p032:cR:e02"].form,
            by_key["torwali2023student:p032:cR:e02"].ipa) == ("b:cā", "b:tʃa:")
    assert "uncertain" in by_key["torwali2023student:p032:cR:e02"].tags
    assert all("�" not in row.form for row in rows)


def test_compiled_cldf_keeps_phonemic_ipa_and_source_citation():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["Source"].startswith("torwali2023student[")
        ]
    assert len(rows) == 1_943
    assert all(row["Phonemic"] and row["Original"] == row["Phonemic"] for row in rows)
    assert all(not row["Native"] for row in rows)
    assert not any("�" in row["Form"] for row in rows)


def test_source_specific_bahrain_and_chail_dialects_are_registered_and_exclusive():
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    bahrain = "dialect:Tor:torwali2023student-BAH:Bahrain%20%28Torwali%202023%29"
    chail = "dialect:Tor:torwali2023student-CHL:Chail%20%28Torwali%202023%29"
    assert dialects["torwali2023student-BAH"]["Tag"] == bahrain
    assert dialects["torwali2023student-CHL"]["Tag"] == chail
    assert dialects["torwali2023student-BAH"]["Language_ID"] == "Tor"
    assert dialects["torwali2023student-CHL"]["Language_ID"] == "Tor"

    form_tags = [row[14].split() for row in read_forms()]
    assert Counter(bahrain in tags for tags in form_tags) == {True: 1_922, False: 21}
    assert Counter(chail in tags for tags in form_tags) == {False: 1_922, True: 21}
    assert all((bahrain in tags) ^ (chail in tags) for tags in form_tags)
    assert not any(any(tag.startswith("dialect:Tor:SSNP-") for tag in tags) for tags in form_tags)


def test_source_pos_labels_are_mapped_to_existing_grammatical_tags():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    installed = {row["Entry_Key"]: row for row in rows if row["Status"] == "installed"}
    expected = {
        "torwali2023student:p013:cR:e03": {"noun"},
        "torwali2023student:p014:cR:e02": {"verb", "tr"},
        "torwali2023student:p018:cL:e01": {"verb", "intr"},
        "torwali2023student:p013:cR:e04": {"adj"},
        "torwali2023student:p013:cL:e02": {"adv"},
        "torwali2023student:p013:cR:e01": {"pron"},
        "torwali2023student:p015:cL:e03": {"noun", "proper-noun"},
        "torwali2023student:p062:cL:e07": {"interj"},
        "torwali2023student:p163:cR:e03": {"conj"},
        "torwali2023student:p048:cL:e04": {"postp"},
        "torwali2023student:p013:cL:e01": {"num"},
        "torwali2023student:p227:cL:e01": {"discourse-marker"},
        "torwali2023student:p156:cR:e01": {"pron", "interr"},
        "torwali2023student:p227:cL:e03": {"pron", "demonstrative"},
        "torwali2023student:p013:cR:e02": {"noun", "pl"},
        # Wrapped definitions exercise the second-baseline POS parser.
        "torwali2023student:p017:cR:e03": {"verb", "tr"},
        "torwali2023student:p052:cL:e04": {"adj"},
        "torwali2023student:p106:cR:e03": {"interj"},
    }
    for key, required in expected.items():
        assert required <= set(installed[key]["Tags"].split())

    grammar_tags = {
        "adj", "adv", "auxiliary", "conj", "demonstrative", "discourse-marker",
        "f", "interj", "interr", "intr", "m", "noun", "num", "ord", "personal",
        "pl", "poss", "postp", "pron", "proper-noun", "quantifier", "relative",
        "sg", "tr", "verb",
    }
    assert all(
        set(row["Tags"].split()) & grammar_tags
        for row in installed.values() if row["Raw_POS"]
    )
    assert not any("unrecognized POS" in row["Reason"] for row in rows)


def test_source_blank_pos_is_documented_without_grammatical_inference():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["Status"] == "installed" and not row["Raw_POS"]
        ]
    assert {row["Entry_Key"] for row in rows} == {
        "torwali2023student:p111:cR:e03",
        "torwali2023student:p111:cL:e04",
        "torwali2023student:p130:cL:e05",
        "torwali2023student:p149:cL:e01",
        "torwali2023student:p165:cR:e02",
        "torwali2023student:p210:cL:e08",
        "torwali2023student:p223:cL:e02",
    }
