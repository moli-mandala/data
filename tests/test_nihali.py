import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
FORMS = ROOT / "data/other/forms"
RAW = FORMS / "raw_data"
SNAPSHOT_SHA256 = "a2525d858969c84eb36c4f5a43857a893b89baa1e0bee16974bc4e8a9d46524d"
FILES = {
    "Mundlay": ("20260817-mundlay-nihali.csv", 1707),
    "Nagaraja": ("20260817-nagaraja-nihali-wiktionary.csv", 1761),
    "Bhattacharya": ("20260817-nihali-database-bhattacharya.csv", 407),
    "Konow": ("20260817-nihali-database-konow.csv", 190),
}


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def read_dicts(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_database_replaces_both_old_imports_and_adds_two_sources():
    all_keys = []
    for _tab, (filename, expected) in FILES.items():
        rows = read_rows(FORMS / filename)
        assert len(rows) == expected
        assert {len(row) for row in rows} == {15}
        assert {row[0] for row in rows} == {"Ni"}
        assert len({row[10] for row in rows}) == len(rows)
        assert all("nihali-database2026[tab " in row[7] for row in rows)
        all_keys.extend(row[10] for row in rows)
    assert len(all_keys) == 4065
    assert len(set(all_keys)) == len(all_keys)


def test_database_audit_is_complete_and_pinned():
    audit = read_dicts(RAW / "20260817-nihali-database-audit.csv")
    assert len(audit) == 4035
    assert {row["Snapshot_SHA256"] for row in audit} == {SNAPSHOT_SHA256}
    assert {row["Snapshot_Exported"] for row in audit} == {"2026-08-17"}
    assert {row["Drive_Modified"] for row in audit} == {
        "2026-04-22T02:38:24.316Z"
    }
    assert Counter((row["Tab"], row["Status"]) for row in audit) == Counter({
        ("Nagaraja", "ingested"): 1696,
        ("Nagaraja", "excluded"): 2,
        ("Mundlay", "ingested"): 1706,
        ("Bhattacharya", "ingested"): 384,
        ("Konow", "ingested"): 190,
        ("Contact", "excluded"): 34,
        ("Roots", "excluded"): 1,
        ("Dravidian", "excluded"): 22,
    })
    assert sum(int(row["Output_Count"] or 0) for row in audit) == 4065
    assert Counter(row["Reason"] for row in audit if row["Status"] == "excluded") == Counter({
        "nonlexical_analysis_sidecar": 45,
        "analysis_sidecar_merged_by_source_id": 12,
        "blank_or_illegible_form": 2,
    })


def test_replacement_key_reconciliation_is_stable_and_conservative():
    key_map = read_dicts(RAW / "20260817-nihali-database-key-map.csv")
    assert len(key_map) == 3468
    assert sum(bool(row["Legacy_Key"]) for row in key_map) == 3104
    assert Counter(row["Reason"] for row in key_map) == Counter({
        "exact_form_gloss": 2919,
        "conservative_fuzzy": 185,
        "unmatched": 364,
    })


def test_representative_records_and_editorial_decisions():
    sources = {
        tab: read_rows(FORMS / filename)
        for tab, (filename, _expected) in FILES.items()
    }
    assert next(row for row in sources["Mundlay"] if row[2] == "dhāblā")[3] == "name of a bird"
    assert next(row for row in sources["Nagaraja"] if row[2] == "bebhum")[3] == "dull (knife, etc.)"
    assert next(row for row in sources["Bhattacharya"] if row[2] == "akhanɖi")[3] == "finger"
    assert next(row for row in sources["Konow"] if row[2] == "ābā-ke")[3] == "to the father"

    exact = next(row for row in sources["Nagaraja"] if row[10] == "nagaraja2014-wiktionary:61")
    assert exact[1] == ">1062"
    assert exact[14] == "loanword"
    uncertain = next(row for row in sources["Nagaraja"] if row[2] == "raccho")
    assert uncertain[1] == ""
    assert "uncertain" in uncertain[14]
    private_use = next(row for row in sources["Nagaraja"] if "\uf0e2" in row[2])
    assert "uncertain" in private_use[14]

    audit = read_dicts(RAW / "20260817-nihali-database-audit.csv")
    private_audit = next(row for row in audit if "\uf0e2" in row["Parsed_Form"])
    assert "private_use_glyph_preserved" in private_audit["Reason"]
