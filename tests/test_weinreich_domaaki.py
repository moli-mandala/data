import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/weinreich_domaaki.py"
SPEC = importlib.util.spec_from_file_location("weinreich_domaaki_extractor", SCRIPT)
weinreich = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = weinreich
SPEC.loader.exec_module(weinreich)


def test_curated_vocabulary_covers_both_varieties_and_all_items():
    assert set(weinreich.ITEMS) == set(range(1, 49))
    assert all({form.lect for form in forms} == {"domaaki_nager", "domaaki_hunza"}
               for forms in weinreich.ITEMS.values())
    assert weinreich.ITEMS[39] == (
        weinreich.N("kirmá", "worm", "3438"),
        weinreich.H("kirmá", "snake", "3438"),
        weinreich.N("jon", "snake", "5110"),
    )


def test_all_curated_turner_links_resolve_to_cdial_entries():
    with (ROOT / "data/cdial/params.csv").open(encoding="utf-8", newline="") as stream:
        valid = {row[0].split(".", 1)[0] for row in csv.reader(stream) if row}
    linked = {
        parameter
        for forms in weinreich.ITEMS.values()
        for form in forms
        for parameter in form.turner_ids
    }
    assert linked
    assert linked <= valid


def test_generated_import_has_stable_rich_schema_and_complete_audit():
    source = ROOT / "data/other/forms/20260813-weinreich-domaaki.csv"
    audit_path = ROOT / "data/other/forms/raw_data/20260813-weinreich-domaaki-audit.csv"
    if not source.exists() or not audit_path.exists():
        return
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    with audit_path.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))

    assert len(rows) == sum(
        max(1, len(form.turner_ids))
        for forms in weinreich.ITEMS.values()
        for form in forms
    )
    assert len(audit) == 48
    assert all(len(row) == weinreich.RICH_COLUMNS for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all(row[7].startswith("weinreich2008[p. ") for row in rows)
    assert all(row[0] in {"domaaki_nager", "domaaki_hunza"} for row in rows)
    assert all(row[14] == "uncertain" for row in rows if row[1])
    assert all("not as a claim about the immediate origin" in row[9] for row in rows if row[1])
    assert not any("\ue466" in value for row in rows for value in row)
    assert [row["Item"] for row in audit] == [f"8.{number}" for number in range(1, 49)]
    assert all(row["Raw_Entry"] for row in audit)


def test_compiled_rows_resolve_to_the_printed_turner_parents():
    source = ROOT / "data/other/forms/20260813-weinreich-domaaki.csv"
    source_keys_path = ROOT / "cldf/form-source-keys.csv"
    aliases_path = ROOT / "cldf/form-id-aliases.csv"
    edges_path = ROOT / "cldf/edges.csv"
    if not all(path.exists() for path in (source, source_keys_path, aliases_path, edges_path)):
        return
    with source.open(encoding="utf-8", newline="") as stream:
        expected = {row[10]: row[1] for row in csv.reader(stream)}
    with source_keys_path.open(encoding="utf-8", newline="") as stream:
        legacy_by_key = {
            row["Source_Key"]: row["Legacy_ID"]
            for row in csv.DictReader(stream)
            if row["Source_Key"].startswith("weinreich-domaaki:")
        }
    with aliases_path.open(encoding="utf-8", newline="") as stream:
        final_by_legacy = {row["Legacy_ID"]: row["Form_ID"] for row in csv.DictReader(stream)}
    with edges_path.open(encoding="utf-8", newline="") as stream:
        parent_by_child = {
            row["Child_ID"]: row["Parent_ID"]
            for row in csv.DictReader(stream)
            if row["Kind"] == "reflex" and row["Rank"] == "1"
        }

    assert legacy_by_key.keys() == expected.keys()
    actual = {
        key: parent_by_child.get(final_by_legacy[legacy_id], "")
        for key, legacy_id in legacy_by_key.items()
    }
    assert actual == expected
