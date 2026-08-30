import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/munda_proto_kherwarian_1968.py"
SPEC = importlib.util.spec_from_file_location("munda_proto_kherwarian_1968_importer", SCRIPT)
source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = source
SPEC.loader.exec_module(source)


def form_rows():
    with source.FORMS.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def param_rows():
    with source.PARAMS.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def audit_rows():
    with source.AUDIT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_complete_source_parameter_and_form_accounting():
    forms = form_rows()
    params = param_rows()
    audit = audit_rows()
    first_by_id = {row["Raw_ID"]: row for row in audit}

    assert len(first_by_id) == 2768
    assert len(params) == 920
    assert len(forms) == len(audit) == 2919
    assert {len(row) for row in forms} == {15}
    assert {len(row) for row in params} == {5}
    assert len({row[0] for row in params}) == 920
    assert len({row[10] for row in forms}) == 2919
    assert Counter(row["Status"] for row in audit) == {"ingested": 2919}
    assert Counter(row["Source_Language"] for row in first_by_id.values()) == {
        "Santali": 925, "pre Mundari": 923, "proto Kherwarian": 920,
    }
    assert Counter(row[0] for row in forms) == {"PKher": 998, "PreMu": 971, "sa": 950}


def test_alignment_is_source_grounded_and_conservative():
    rows = audit_rows()
    first = {row["Raw_ID"]: row for row in rows}
    methods = Counter(row["Alignment_Method"] for row in first.values())
    assert methods == {
        "locator+gloss": 1832, "self": 920, "unique-gloss": 11,
        "curated-form-meaning": 2, "unlinked": 3,
    }

    direct = first["munda1968proto:C:c1.p385.i80"]
    assert (direct["Parameter_ID"], direct["Alignment_Method"]) == (
        "pkh-385-80", "locator+gloss"
    )
    recovered = first["munda1968proto:R:c2.p37.i53"]
    assert (recovered["Parameter_ID"], recovered["Alignment_Method"]) == (
        "pkh-274-71", "unique-gloss"
    )
    curated = first["munda1968proto:C:c1.p.i51"]
    assert (curated["Parameter_ID"], curated["Alignment_Method"]) == (
        "pkh-1-51", "curated-form-meaning"
    )
    unlinked = {
        row["Raw_ID"] for row in first.values() if row["Alignment_Method"] == "unlinked"
    }
    assert unlinked == {
        "munda1968proto:C:c1.p1002.i126-1",
        "munda1968proto:C:c1.p202.i65",
        "munda1968proto:R:c2.p202.i65",
    }
    assert all(not first[raw_id]["Parameter_ID"] for raw_id in unlinked)


def test_variant_structure_gloss_cleanup_and_tags():
    rows = audit_rows()
    three = [row for row in rows if row["Raw_ID"] == "munda1968proto:R:c3.p676.i102"]
    assert [row["Final_Form"] for row in three] == ["*(a-)pì/ɛ̀", "*a-pɛ̀-a"]
    assert three[1]["Variant_Of_Key"] == three[0]["Entry_Key"]
    assert source.split_variants("*ajur ~ *ajar") == ["*ajur", "*ajar"]
    assert source.split_variants("*(a-)pì/ɛ̀") == ["*(a-)pì/ɛ̀"]

    pronoun = next(row for row in rows if row["Raw_ID"] == "munda1968proto:R:c3.p482.i88")
    assert pronoun["Final_Gloss"] == "you two, 2nd person dual"
    assert set(pronoun["Tags"].split()) == {"pron", "du"}
    scientific = next(row for row in rows if "{Sci.name}" in row["Raw_Gloss"])
    assert scientific["Final_Gloss"] == "a certain tree (Bauhinia variegata)"
    assert all("{Sci.name}" not in row[3] for row in form_rows())


def test_manifest_profile_languages_and_review_sample():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 2768
    assert manifest["parameter_rows"] == 920
    assert manifest["form_rows"] == 2919
    assert manifest["excluded_rows"] == 0
    assert manifest["policy"]["extraction"] == "structured HTML semantic spans; no OCR"

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as handle:
        languages = {row["ID"]: row for row in csv.DictReader(handle)}
    assert languages["PKher"]["Name"] == "Proto-Kherwarian"
    assert languages["PreMu"]["Name"] == "Pre-Mundari"
    assert languages["PKher"]["Clade"] == languages["PreMu"]["Clade"] == "Munda"
    assert not languages["PKher"]["Glottocode"] and not languages["PreMu"]["Glottocode"]

    profile = source.PROFILE.read_text(encoding="utf-8")
    assert profile.startswith("Grapheme\tIPA\n \t#\n")
    assert "�" not in profile
    assert all("�" not in "".join(row) for row in form_rows())

    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == len({row["Raw_ID"] for row in sample}) == 20
    assert {row["Review_Result"] for row in sample} == {"pass"}
    assert {row["Material_Error"] for row in sample} == {""}


def test_offline_rebuild_is_exact():
    params, forms, audit = source.transform(source.offline_records())
    assert params == param_rows()
    assert forms == form_rows()
    assert audit == audit_rows()


def test_compiled_rows_survive_when_current_cldf_is_built():
    compiled = ROOT / "cldf/forms.csv"
    if not compiled.exists():
        return
    with compiled.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if "munda1968proto" in row["Source"] and row.get("Status") != "entry"
        ]
    if not rows:
        return
    assert len(rows) == 2919
    assert {row["Language_ID"] for row in rows} == {"PKher", "PreMu", "sa"}
    assert all(row["Original"] == row["Form"] == row["Phonemic"] for row in rows)
    assert sum(not row["Parameter_ID"] for row in rows) == 3
