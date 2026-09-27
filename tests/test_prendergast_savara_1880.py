"""Source-stage checks for Prendergast's complete 1880/1881 Savara vocabulary."""

import csv
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/prendergast_savara_1880"
INSTALLED = DATA / "data/other/forms/20260925-prendergast-savara.csv"
PROFILE = DATA / "conversion/prendergast-savara-1880.txt"
spec = importlib.util.spec_from_file_location("prendergast_savara", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_complete_source_and_reproducible_rows():
    rows, audit = source.build()
    with (PACKAGE / "20260925-prendergast-savara.csv").open(encoding="utf-8", newline="") as handle:
        staged = list(csv.reader(handle))
    assert rows == staged
    with INSTALLED.open(encoding="utf-8", newline="") as handle:
        assert rows == list(csv.reader(handle))
    assert len(audit) == 266 and len(rows) == 270
    stored_audit = [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]
    assert audit == stored_audit
    assert Counter(a["status"] for a in audit) == {"selected": 266}
    assert Counter((a["printed_page"], a["column"]) for a in audit) == source.EXPECTED_CELLS
    assert len({r[10] for r in rows}) == 270
    assert all(len(r) == 15 and r[0] == "so" and r[14].startswith(source.DIALECT) for r in rows)
    assert all(r[7].startswith("prendergast1881savara[p. ") for r in rows)
    assert {r[3] for r in rows} >= {"good", "bad", "brass pot", "big stick", "little stick"}
    assert all(not r[6] and not r[9] and not r[12] and not r[13] for r in rows)
    assert sum(bool(r[11]) for r in rows) == 4
    assert {r[10] for r in rows if "uncertain" in r[14]} == {
        "prendergast1881savara:p426:left:13", "prendergast1881savara:p426:left:28"}
    assert not any("(" in r[2] or ")" in r[2] for r in rows)


def test_print_review_source_provenance_and_overlap():
    _, audit = source.build()
    review = [json.loads(line) for line in (PACKAGE / "visual-review-20260925.jsonl").read_text().splitlines()]
    assert len(review) == 59
    assert {r["entry_key"] for r in review} == {a["entry_key"] for a in audit[:59]}
    assert all(r["print_cell_reviewed"] and not r["accepted_form_material_error_remaining"] for r in review)
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["append_order"] == 108
    assert tuple(manifest[k] for k in ("printed_prompt_cells", "selected_prompt_cells", "held_prompt_cells", "installed_forms_planned")) == (266, 266, 0, 270)
    assert "public domain" in manifest["rights"].lower()
    assert all(name in manifest["provenance"] for name in ("Prendergast", "Cain", "Cust"))
    assert "blank coordinates" in manifest["provenance"]
    assert len(manifest["scan_sha256"]) == len(manifest["rendered_page_426_sha256"]) == 64
    overlap = json.loads((PACKAGE / "overlap-review.json").read_text())
    assert overlap["existing_Sora_source_rows_checked"] == 2171
    assert overlap["selected_forms_checked"] == 270
    assert len(overlap["exact_matches"]) == 12
    assert {r["entry_key"] for r in overlap["exact_matches"]} >= {
        "prendergast1881savara:p426:left:03",
        "prendergast1881savara:p426:left:26",
        "prendergast1881savara:p426:left:27",
        "prendergast1881savara:p427:right:05",
    }
    assert overlap["diacritic_insensitive_additional_match"]["item"] == 44
    assert not overlap["same_source_reuse_found"]


def test_registered_dialect_bibliography_profile_and_scoped_parse():
    rows, _ = source.build()
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open(encoding="utf-8", newline=""))}
    d = dialects["prendergast-1880-savara"]
    assert d[1:5] == [source.DIALECT, "so", "prendergast1881savara:Savara", "Savara (Prendergast 1880)"]
    assert d[5:8] == ["", "", ""]
    bib = (DATA / "cldf/sources.bib").read_text(encoding="utf-8")
    assert bib.count("@article{prendergast1881savara,") == 1
    assert "20260925-prendergast-savara.csv" in bib
    tokenizer = Tokenizer(str(PROFILE))
    for row in rows:
        assert "�" not in tokenizer(row[2], column="IPA")
    assert tokenizer("gochang", column="IPA").replace(" ", "") == "gochang"
    assert tokenizer("shenḍātāng", column="IPA").replace(" ", "") == "shenḍātāng"
    import sys
    sys.path.insert(0, str(DATA))
    import make_cldf
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(INSTALLED), errors, name=INSTALLED.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 270
    assert parsed[0].old_form == parsed[0].form == "mingnyan"
    assert not parsed[0].notes
