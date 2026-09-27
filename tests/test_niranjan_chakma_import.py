"""Source-only importer contracts. These tests never invoke a database build."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/niranjan_chakma_2010"


def importer():
    spec = importlib.util.spec_from_file_location("niranjan_chakma_import", ROOT / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_accounting_and_script_preservation():
    rows, audit = importer().prepare()
    assert len(audit) == 733 and len(rows) == 742
    assert sum(r["status"] == "withheld" for r in audit) == 2
    assert sum(len(r["emitted_rows"]) for r in audit) == len(rows)
    assert sum(bool(r[14]) for r in rows) == 97
    assert all(len(r) == 15 and r[0] == "Chakma" for r in rows)
    assert all(r[2] == r[4] and not r[5] for r in rows)
    assert all(not any(r[i] for i in (1, 8, 9, 11, 12, 13)) for r in rows)
    assert len({r[10] for r in rows}) == len(rows)
    assert all("candidate" in r and "review" in r for r in audit)


def test_multiple_forms_and_withheld_source_cells():
    _, audit = importer().prepare()
    records = {(r["locator"]["pdf_page"], r["locator"]["candidate_row"]): r for r in audit}
    flood = records[22, 30]
    assert [r[2] for r in flood["emitted_rows"]] == ["বান", "পানিবান"]
    assert all(not r[11] for r in flood["emitted_rows"])
    assert records[38, 16]["status"] == "withheld"
    assert records[38, 16]["uncertainty_types"] == ["segmentation"]
    assert records[23, 16]["status"] == "withheld"
    sprout = records[24, 14]
    assert sprout["review"]["reviewed_english"] == "Spout"
    assert sprout["emitted_rows"][0][3] == "sprout"
    assert "gloss" in sprout["uncertainty_types"]
    assert sprout["emitted_rows"][0][14] == "uncertain"
    assert records[35, 3]["emitted_rows"][0][10] != records[35, 19]["emitted_rows"][0][10]


def test_keys_survive_review_order_and_spelling_corrections(monkeypatch):
    module = importer()
    before, _ = module.prepare()
    original = module.read_jsonl

    def changed(name):
        records = original(name)
        if name == "visual-review.jsonl":
            target = next(r for r in records if (r["pdf_page"], r["candidate_row"]) == (21, 20))
            target["reviewed_chakma"] += "া"
            target["reviewed_english"] = "Corrected gloss"
            records.reverse()
        return records

    monkeypatch.setattr(module, "read_jsonl", changed)
    after, _ = module.prepare()
    assert [r[10] for r in before] == [r[10] for r in after]
    assert sum(a != b for a, b in zip(before, after)) == 1


def test_complete_inventory_and_draft_preservation_settings():
    import json
    import unicodedata
    from collections import Counter
    from source_meta import SourceMeta
    from pybtex.database import parse_file
    rows, _ = importer().prepare()
    inventory = json.loads((ROOT / "symbol-inventory.json").read_text())
    assert {r["character"]: r["count"] for r in inventory["symbols"]} == Counter(ch for r in rows for ch in r[2])
    assert all(unicodedata.is_normalized("NFC", r[2]) for r in rows)
    assert all(0x0980 <= ord(ch) <= 0x09FF or ch in " -’" for r in rows for ch in r[2])
    settings = SourceMeta([ROOT / "20260921-niranjan-chakma.yaml"])
    defaults = settings.files["20260921-niranjan-chakma"]
    assert defaults["transcription"] == [{"convert": False, "preserve_hyphens": True}]
    source = settings.sources["niranjan2010chakma"]
    assert source["identity"]["dedupe_by_entry_key"]
    assert source["forms"]["split_alternates"] is False
    bibliography = parse_file(str(ROOT / "source.bib"))
    assert set(bibliography.entries) == {"niranjan2010chakma"}
    assert all(r[7].startswith("niranjan2010chakma[") for r in rows)


def test_installed_source_and_lightweight_parser_preserve_spelling():
    import csv
    import io
    from make_cldf import parse_file
    from pybtex.database import parse_file as parse_bib
    module = importer()
    expected, _ = module.prepare()
    forms = ROOT.parent.parent
    path = forms / f"{module.STEM}.csv"
    with path.open() as stream:
        assert list(csv.reader(stream)) == expected
    errors = io.StringIO()
    parsed, stats = parse_file(str(path), errors=errors)
    assert not errors.getvalue()
    assert len(parsed) == 742 and stats == {"converted": 0, "for_conversion": 0}
    originals = {row[10]: row for row in expected}
    for row in parsed:
        assert row.form == row.old_form == originals[row.entry_key][2]
        assert not row.ipa
    repo = Path(__file__).resolve().parents[1]
    assert parse_bib(str(repo / "cldf/sources.bib")).entries["niranjan2010chakma"] == parse_bib(str(ROOT / "source.bib")).entries["niranjan2010chakma"]
    with (repo / "cldf/languages.csv").open() as stream:
        language = next(r for r in csv.DictReader(stream) if r["ID"] == "Chakma")
    assert language["Clade"] == "Eastern"
