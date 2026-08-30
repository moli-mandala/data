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
    # Chhattisgarhi and Rajasthani now keep their curated Parameter_ID values in the
    # installed survey CSVs.  Their duplicate, positional-ID restoration records were
    # deliberately removed; Kannauji and the other durable-ID restorations remain here.
    # 14,506 self-links were also dropped.  Legacy modelled a headword as two rows — the entry
    # plus an attested row beneath it — and the edge model collapses both onto one node, so the
    # importer resolved that pair into a link from the node to itself.  Installing those made
    # 14,506 CDIAL headwords their own etymon and dropped them from the entry list.
    assert len(restored) == 4_580

    with (ROOT / "cldf/edges.csv").open(encoding="utf-8") as handle:
        accepted = {
            (row["Child_ID"], row["Parent_ID"])
            for row in csv.DictReader(handle)
            if row["Rank"] == "1"
        }
    assert restored <= accepted
    assert ("f_4y2f5moveah5a", "12335") in restored  # Kannauji sarir ‘body’


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
        "bagri": 1_921,
        "chattisgarhi": 2_079,
        "dhundari": 1_264,
        "hadothi": 1_691,
        "kannauji": 1_991,
        "marwari": 1_880,
        "mewari": 3_368,
        "mewati": 1_426,
    }
    for reference, count in expected.items():
        form_ids = {
            row["ID"] for row in forms if reference in citation_keys(row.get("Source", ""))
        }
        assert len(form_ids & linked) == count

    # Rau's 127 Proto-Munda forms used to appear here, but all 127 of their restored links were
    # self-links: legacy held the etymon (`m1`) and its head-form row (`3-0`) separately, and
    # this model has one node for both.  They are the etyma, so they carry no rank-1 edge.
    rau_ids = {row["ID"] for row in forms if "rau" in citation_keys(row.get("Source", ""))}
    assert len(rau_ids) == 127
    assert not rau_ids & linked
    assert {row["Status"] for row in forms if row["ID"] in rau_ids} == {"entry"}
