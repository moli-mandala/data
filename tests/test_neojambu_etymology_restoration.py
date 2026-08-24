import csv
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def citation_keys(value):
    return {
        part.strip().split("[", 1)[0]
        for part in (value or "").split(";")
        if part.strip()
    }


def test_legacy_etymology_restoration_is_installed_and_audited():
    summary_path = (
        ROOT / "data/other/forms/raw_data/20260820-neojambu-etymology-restoration-summary.json"
    )
    audit_path = (
        ROOT / "data/other/forms/raw_data/20260820-neojambu-etymology-restoration-audit.csv.gz"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["legacy_database_sha256"] == (
        "9d02428a49cb96eb0e79d90d2a2918f324c275258c0b3fdc51ef4b989728d3d0"
    )
    assert summary["legacy_link_rows"] == 290_071
    assert summary["unique_restored_edges"] == 32_635
    assert summary["restored_assignment_rows"] == 32_635
    assert summary["reference_counts"]["chattisgarhi"] == {"restored": 2_111}
    assert summary["reference_counts"]["kannauji"] == {
        "already-present": 9,
        "restored": 1_976,
        "unresolved-child": 169,
    }
    assert summary["reference_counts"]["rau"]["restored"] == 127
    assert sum(
        summary["reference_counts"][reference]["restored"]
        for reference in {"mewari", "hadothi", "dhundari", "marwari", "mewati", "bagri"}
    ) == 11_739

    with gzip.open(audit_path, "rt", encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == summary["audit_rows"] == 290_300
    assert {row["Status"] for row in audit} == {
        "already-present",
        "current-link-preserved",
        "legacy-merge-conflict",
        "restored",
        "unresolved-child",
        "unresolved-etymon",
    }


def test_restored_assignments_survive_in_the_compiled_edge_graph():
    with (ROOT / "data/etymology-assignments.csv").open(encoding="utf-8") as handle:
        restored = {
            (row["Form_ID"], row["Etymon_ID"])
            for row in csv.DictReader(handle)
            if row["Notes"] == "Restored from legacy NeoJambu origin_lemma_id"
        }
    assert len(restored) == 32_635

    with (ROOT / "cldf/edges.csv").open(encoding="utf-8") as handle:
        accepted = {
            (row["Child_ID"], row["Parent_ID"])
            for row in csv.DictReader(handle)
            if row["Rank"] == "1"
        }
    assert restored <= accepted
    assert ("f_37lopoca7p77o", "992") in restored  # Bagri hū ‘I’


def test_previously_lost_source_groups_have_accepted_edges():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8") as handle:
        forms = list(csv.DictReader(handle))
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8") as handle:
        linked = {
            row["Child_ID"]
            for row in csv.DictReader(handle)
            if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
        }
    expected = {
        "bagri": 1_937,
        "chattisgarhi": 2_111,
        "dhundari": 1_282,
        "hadothi": 1_742,
        "kannauji": 1_985,
        "marwari": 1_905,
        "mewari": 3_428,
        "mewati": 1_443,
        "rau": 127,
    }
    for reference, count in expected.items():
        form_ids = {
            row["ID"] for row in forms if reference in citation_keys(row.get("Source", ""))
        }
        assert len(form_ids & linked) == count
