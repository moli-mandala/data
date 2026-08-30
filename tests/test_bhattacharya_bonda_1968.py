import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/bhattacharya_bonda_1968.py"
SPEC = importlib.util.spec_from_file_location("bhattacharya_bonda_1968_importer", SCRIPT)
source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = source
SPEC.loader.exec_module(source)


def installed_rows():
    with source.OUTPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def audit_rows():
    with source.AUDIT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_audit_rows():
    first = {}
    for row in audit_rows():
        first.setdefault(row["Raw_ID"], row)
    return first


def test_complete_source_alternant_and_exclusion_accounting():
    installed = installed_rows()
    audit = audit_rows()
    first = first_audit_rows()
    assert len(first) == 2881
    assert len(audit) == 3331
    assert len(installed) == 3330
    assert {len(row) for row in installed} == {15}
    assert len({row[10] for row in installed}) == 3330
    assert Counter(row["Status"] for row in audit) == {
        "ingested": 3330, "excluded": 1,
    }
    assert Counter(row["Source_Language"] for row in first.values()) == {
        "Bondo [Plains]": 2716, "Bondo [Hill]": 165,
    }
    assert Counter(row[14].split()[0] for row in installed) == {
        source.LANGUAGE_MAP["Bondo [Plains]"]: 3146,
        source.LANGUAGE_MAP["Bondo [Hill]"]: 184,
    }
    excluded = next(row for row in audit if row["Status"] == "excluded")
    assert excluded["Raw_ID"] == "bhattacharya1968bonda:C:c1.p77.r4.i1531.s1526"
    assert excluded["Final_Form"] == "da?tukui"
    assert excluded["Reason"] == "exact repeated alternant inside one source record"


def test_cross_references_are_resolved_only_from_printed_targets():
    first = first_audit_rows()
    statuses = Counter(row["Crossref_Status"] for row in first.values())
    assert statuses == {
        "not-cross-reference": 2845,
        "resolved": 27,
        "resolved-gloss-multiple-targets": 1,
        "unresolved": 8,
    }

    bobo = first["bhattacharya1968bonda:C:c1.p100.r10.i2007.s2002"]
    assert bobo["Raw_Gloss"] == "see <babu>"
    assert bobo["Final_Gloss"] == "a term used to address younger ones endearingly"
    assert bobo["Crossref_Target_Key"] == (
        "bhattacharya1968bonda:C:c1.p92.r11.i1848.s1843"
    )
    assert bobo["Variant_Of_Key"] == bobo["Crossref_Target_Key"]

    homograph = first["bhattacharya1968bonda:C:c1.p62.r20.i1264.s1258"]
    assert homograph["Raw_Gloss"] == "see <ḍem-> to be"
    assert homograph["Final_Gloss"] == "to be"
    assert homograph["Crossref_Target_Key"].endswith("p63.r4.i1270.s1264")

    multiple = first["bhattacharya1968bonda:C:c1.p30.r16.i591.s585"]
    assert multiple["Final_Gloss"] == "to abuse"
    assert multiple["Crossref_Status"] == "resolved-gloss-multiple-targets"
    assert not multiple["Crossref_Target_Key"]
    assert not multiple["Variant_Of_Key"]

    unresolved = first["bhattacharya1968bonda:C:c1.p94.r5.i1887.s1882"]
    assert unresolved["Raw_Gloss"] == "see <raŋbip'>"
    assert unresolved["Crossref_Status"] == "unresolved"
    assert not unresolved["Final_Gloss"]
    assert not unresolved["Variant_Of_Key"]

    keys = {row[10] for row in installed_rows()}
    assert all(not row[11] or row[11] in keys for row in installed_rows())


def test_notes_etymology_uncertainty_and_question_mark_policy():
    first = first_audit_rows()
    assert Counter(row["Note_Class"] for row in first.values()) == {
        "none": 1115, "comment": 1194, "etymology": 550, "query": 22,
    }
    etymology = first["bhattacharya1968bonda:C:c1.p1.r4.i4.s4"]
    assert etymology["Final_Etymology"] == "De./ha:t/"
    assert not etymology["Final_Notes"]

    split_note = first["bhattacharya1968bonda:C:c1.p2.r11.i35.s35"]
    assert split_note["Final_Etymology"] == "De."
    assert "/laǰ-abur nela-rem/" in split_note["Final_Notes"]

    ordinary_question_mark = first["bhattacharya1968bonda:C:c1.p5.r3.i87.s85"]
    assert "uncertain" not in ordinary_question_mark["Tags"].split()
    assert "?" not in ordinary_question_mark["Raw_Form"]
    glottal_symbol = next(
        row for row in first.values()
        if "?" in row["Raw_Form"] and row["Note_Class"] == "none"
        and not row["Raw_Form"].endswith("(E?)")
    )
    assert "uncertain" not in glottal_symbol["Tags"].split()

    no_gloss = first["bhattacharya1968bonda:C:c2.p142.r13.i2813.s"]
    assert no_gloss["Raw_Form"] == "gige(E?)"
    assert no_gloss["Final_Form"] == "gige"
    assert not no_gloss["Final_Gloss"]
    assert "uncertain" in no_gloss["Tags"].split()


def test_variants_dialect_registry_manifest_profile_and_sample():
    assert source.split_variants("a, b (c, d); e") == ["a", "b (c, d)", "e"]
    rows = [
        row for row in audit_rows()
        if row["Raw_ID"] == "bhattacharya1968bonda:C:c1.p140.r6.i2788.s2787"
    ]
    assert [row["Final_Form"] for row in rows] == ["'ɔ:ɔ:", "'ɔn"]
    assert rows[1]["Variant_Of_Key"] == rows[0]["Entry_Key"]

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        dialects = {row["ID"]: row for row in csv.DictReader(handle)}
    for dialect_id, name in [
        ("bhattacharya1968bonda-plains", "Plains Bondo"),
        ("bhattacharya1968bonda-hill", "Hill Bondo"),
    ]:
        row = dialects[dialect_id]
        assert row["Language_ID"] == "re"
        assert row["Name"] == name
        assert row["Glottocode"] == "bond1245"
        assert not row["Latitude"] and not row["Longitude"] and not row["Quality"]
        assert "supplies no locality" in row["Location"]

    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 2881
    assert manifest["source_variant_rows"] == 3331
    assert manifest["installed_rows"] == 3330
    assert manifest["excluded_rows"] == 1
    assert manifest["policy"]["extraction"].endswith("no OCR")

    profile = source.PROFILE.read_text(encoding="utf-8")
    assert profile.startswith("Grapheme\tIPA\n \t#\n")
    assert "�" not in profile
    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == len({row["Raw_ID"] for row in sample}) == 20
    assert sum(row["Source_Language"] == "Bondo [Hill]" for row in sample) >= 4
    assert sum(row["Note_Class"] == "etymology" for row in sample) >= 4
    assert {row["Review_Result"] for row in sample} == {"pass"}
    assert {row["Material_Error"] for row in sample} == {""}


def test_offline_rebuild_is_exact():
    installed, audit = source.transform(source.offline_records())
    assert installed == installed_rows()
    assert audit == audit_rows()


def test_compiled_rows_survive_when_current_cldf_is_built():
    path = ROOT / "cldf/forms.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if SOURCE_KEY in row["Source"]]
    if not rows:
        return
    assert len(rows) == 3330
    assert {row["Language_ID"] for row in rows} == {"re"}
    assert all(not row["Parameter_ID"] for row in rows)
    assert all(row["Original"] == row["Form"] == row["Phonemic"] for row in rows)

    reference_path = ROOT / "cldf/references.csv"
    if reference_path.exists():
        with reference_path.open(encoding="utf-8", newline="") as handle:
            references = {row["ID"]: row for row in csv.DictReader(handle)}
        reference = references[SOURCE_KEY]
        assert "Deccan College Postgraduate and Research Institute" in reference["Source"]
        assert reference["OCR"] == "No"
        assert reference["Etymology_Provenance"] == "source"


SOURCE_KEY = source.SOURCE_KEY
