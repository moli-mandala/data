import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/pinnow_munda_1959.py"
SPEC = importlib.util.spec_from_file_location("pinnow_munda_1959_importer", SCRIPT)
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


def test_exact_source_expansion_and_complete_audit():
    installed = installed_rows()
    audit = audit_rows()
    assert len({row["Raw_ID"] for row in audit}) == 3340
    assert len(audit) == 4053
    assert len(installed) == 4051
    assert {len(row) for row in installed} == {15}
    assert len({row[10] for row in installed}) == 4051
    assert Counter(row["Status"] for row in audit) == {"ingested": 4051, "excluded": 2}
    assert Counter(row["Link_Status"] for row in audit) == {
        "unlinked": 3126,
        "direct": 905,
        "disambiguated": 16,
        "ambiguous": 3,
        "no-set": 1,
        "not-applicable": 2,
    }


def test_source_languages_variants_and_missing_record_are_accounted_for():
    audit = audit_rows()
    first_by_id = {row["Raw_ID"]: row for row in audit}
    assert Counter(row["Source_Language"] for row in first_by_id.values()) == {
        "Mundari": 643, "Santali": 551, "Kharia": 545, "Sora": 439,
        "Ho": 272, "Birhor": 265, "Korku": 214, "Bodo-Gadaba": 150,
        "Bondo": 65, "Juang": 63, "Mahali": 60, "Korwa": 48,
        "Asuri": 19, "Turi": 6,
    }
    missing = first_by_id[
        "pinnow1959versuch:C:c5.p174.i1619-2.sV381"
    ]
    assert (missing["Raw_Form"], missing["Status"], missing["Reason"]) == (
        "MISSING", "excluded", "source explicitly marks the form MISSING"
    )

    variants = [
        row for row in audit
        if row["Raw_ID"] == "pinnow1959versuch:C:c1.p272.i506.sK370"
    ]
    assert [row["Final_Form"] for row in variants] == ["ɟhɛntu", "ɟɛntu", "ɟintu"]
    assert variants[1]["Variant_Of_Key"] == variants[0]["Entry_Key"]


def test_top_level_variant_split_preserves_parenthetical_source_notation():
    assert source.split_variants("biro, buroŋ") == ["biro", "buroŋ"]
    assert source.split_variants("ueːdɑː-n (? uːl-dɑː-)") == ["ueːdɑː-n (? uːl-dɑː-)"]
    assert source.split_variants("a (b, c), d") == ["a (b, c)", "d"]


def test_proto_munda_links_follow_printed_cross_references_conservatively():
    audit = audit_rows()
    ear = next(
        row for row in audit
        if row["Raw_ID"] == "pinnow1959versuch:C:c3.p97.i1319.sV147"
    )
    assert (ear["Pinnow_Set"], ear["Parameter_ID"], ear["Link_Status"]) == (
        "V147", "m73", "direct"
    )
    assert ear["Citation"] == "pinnow1959versuch[p. 97, item 1319, set V147]"

    v3 = [row for row in audit if row["Pinnow_Set"] == "V3"]
    assert len(v3) == 9
    assert {(row["Parameter_ID"], row["Link_Status"]) for row in v3} == {
        ("m76", "disambiguated")
    }

    v278 = [row for row in audit if row["Pinnow_Set"] == "V278"]
    assert len(v278) == 10
    assert Counter(row["Link_Status"] for row in v278) == {
        "disambiguated": 7, "ambiguous": 3
    }
    assert all(not row["Parameter_ID"] for row in v278 if row["Link_Status"] == "ambiguous")

    empty = next(row for row in audit if row["Raw_ID"].endswith(".sEMPTY"))
    assert (empty["Pinnow_Set"], empty["Parameter_ID"], empty["Link_Status"]) == (
        "", "", "no-set"
    )


def test_manifest_profile_and_new_languages():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 3340
    assert manifest["installed_rows"] == 4051
    assert manifest["pinnow_numbered_sets"] == 552
    assert manifest["source_set_labels"] == 553
    assert manifest["policy"]["extraction"] == "structured HTML semantic spans; no OCR"

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as handle:
        languages = {row["ID"]: row for row in csv.DictReader(handle)}
    assert languages["Asuri"]["Glottocode"] == "asur1254"
    assert languages["Birhor"]["Glottocode"] == "birh1242"
    assert languages["Turi"]["Glottocode"] == "turi1246"
    assert {languages[key]["Clade"] for key in ("Asuri", "Birhor", "Turi")} == {"Munda"}

    profile = source.PROFILE.read_text(encoding="utf-8")
    assert profile.startswith("Grapheme\tIPA\n \t#\n")
    assert "�" not in profile
    assert all("�" not in "".join(row) for row in installed_rows())


def test_seeded_review_sample_and_offline_rebuild():
    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == 20
    assert len({row["Raw_ID"] for row in sample}) == 20
    assert {row["Review_Result"] for row in sample} == {"pass"}
    assert {row["Material_Error"] for row in sample} == {""}
    assert len({row["Source_Language"] for row in sample}) >= 7

    offline_installed, offline_audit = source.transform(source.offline_records())
    assert offline_installed == installed_rows()
    assert offline_audit == audit_rows()


def test_compiled_rows_survive_with_identity_and_phonemic_layers():
    compiled_path = ROOT / "cldf/forms.csv"
    if not compiled_path.exists():
        return
    with compiled_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if "pinnow1959versuch" in row["Source"] and row["Status"] != "entry"
        ]
    if not rows:
        return
    assert len(rows) == 4051
    assert all(row["Original"] == row["Form"] == row["Phonemic"] for row in rows)
    assert {"Asuri", "Birhor", "Turi"} <= {row["Language_ID"] for row in rows}

    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as handle:
        keys = [
            row for row in csv.DictReader(handle)
            if row["Source_Key"].startswith("pinnow1959versuch:")
        ]
    assert len(keys) == 4051
    assert len({row["Source_Key"] for row in keys}) == 4051

    ids = {row["ID"] for row in rows}
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8", newline="") as handle:
        edges = [
            row for row in csv.DictReader(handle)
            if row["Child_ID"] in ids and row["Rank"] == "1"
        ]
    assert Counter(row["Kind"] for row in edges) == {"reflex": 778, "variant": 712}
    assert len({row["Parent_ID"] for row in edges if row["Kind"] == "reflex"}) == 92
