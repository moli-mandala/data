import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/zide_sora_gorum_1982.py"
SPEC = importlib.util.spec_from_file_location("zide_sora_gorum_1982_importer", SCRIPT)
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


def test_complete_source_and_alternant_accounting():
    installed = installed_rows()
    audit = audit_rows()
    first = {row["Raw_ID"]: row for row in audit}
    assert len(first) == 1750
    assert len(installed) == len(audit) == 2057
    assert {len(row) for row in installed} == {15}
    assert len({row[10] for row in installed}) == 2057
    assert Counter(row["Status"] for row in audit) == {"ingested": 2057}
    assert Counter(row["Source_Language"] for row in first.values()) == {
        "Sora": 953, "Juray": 797,
    }
    assert Counter(row[0] for row in installed) == {"so": 1138, "Juray": 919}
    assert all(not row[1] for row in installed)


def test_source_comparison_groups_are_preserved_without_proto_links():
    first = {row["Raw_ID"]: row for row in audit_rows()}
    groups = {row["Comparison_Group"]: row["Group_Status"] for row in first.values()}
    assert len(groups) == 1011
    assert Counter(groups.values()) == {"paired": 739, "singleton": 272}

    sora = first["zide1982reconstruction:C:c1.p461.i1462"]
    juray = first["zide1982reconstruction:C:c2.p461.i1462"]
    assert sora["Comparison_Group"] == juray["Comparison_Group"] == "Z82-p461-i1462"
    assert sora["Group_Status"] == juray["Group_Status"] == "paired"
    assert all(not row[1] and row[8].startswith("Z82-") for row in installed_rows())
    assert all("No protoform is exposed" in row[9] for row in installed_rows())


def test_variants_parenthetical_punctuation_and_grammar_tags():
    variants = [
        row for row in audit_rows()
        if row["Raw_ID"] == "zide1982reconstruction:C:c1.p309.i524"
    ]
    assert [row["Final_Form"] for row in variants] == ["nʌmi", "nʌmɨɟ", "nam", "nʌm", "lam"]
    assert all(row["Variant_Of_Key"] == variants[0]["Entry_Key"] for row in variants[1:])
    assert source.split_variants("a, b (c, d); e") == ["a", "b (c, d)", "e"]

    prefix = next(row for row in audit_rows() if row["Raw_Gloss"].endswith("(prefix)"))
    assert "prefix" in prefix["Tags"].split()
    assert not prefix["Final_Gloss"].endswith("(prefix)")


def test_manifest_profile_language_registry_and_sample():
    manifest = json.loads(source.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == source.SOURCE_SHA256
    assert manifest["source_records"] == 1750
    assert manifest["installed_rows"] == 2057
    assert manifest["comparison_groups"] == 1011
    assert manifest["excluded_rows"] == 0
    assert manifest["policy"]["extraction"] == "structured HTML semantic spans; no OCR"

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as handle:
        languages = {row["ID"]: row for row in csv.DictReader(handle)}
    assert languages["Juray"]["Glottocode"] == "jura1242"
    assert languages["Juray"]["Clade"] == "Munda"
    assert languages["Juray"]["Quality"] == "C"
    assert languages["ju"]["Name"] == "Juang"

    profile = source.PROFILE.read_text(encoding="utf-8")
    assert profile.startswith("Grapheme\tIPA\n \t#\n")
    assert "�" not in profile
    with source.SAMPLE.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == len({row["Raw_ID"] for row in sample}) == 20
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
        rows = [row for row in csv.DictReader(handle) if "zide1982reconstruction" in row["Source"]]
    if not rows:
        return
    assert len(rows) == 2057
    assert {row["Language_ID"] for row in rows} == {"so", "Juray"}
    assert all(not row["Parameter_ID"] for row in rows)
    assert all(row["Original"] == row["Form"] == row["Phonemic"] for row in rows)
