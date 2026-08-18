import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/nured_org.py"
SPEC = importlib.util.spec_from_file_location("nured_org_importer", SCRIPT)
nured = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = nured
SPEC.loader.exec_module(nured)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_head_normalization_removes_accents_but_preserves_length_and_segments():
    assert nured.normalize_head("*mā́lā-") == nured.normalize_head("mālā")
    assert nured.normalize_head("mā́lā") != nured.normalize_head("mala")
    assert nured.normalize_head("aṅkurá-") != nured.normalize_head("ankura-")
    assert nured.head_variants("*ankra(ka)-") == {"ankra", "ankraka"}


def test_printed_turner_ids_are_validated_and_deduplicated():
    text = "See {{CDIAL}} (T. 3906), T. 3906, and an invalid T. 999999."
    assert nured.printed_turner_ids(text, {"3906", "1"}) == ["3906"]


def test_named_template_arguments_and_source_order_do_not_hide_the_headword():
    text = (
        "{{Lemma|lang=x-pnur|*āč[iya]n[a]-|n. m.|woolen thread}}\n"
        "== Etymology of Source ==\n"
        "* '''Old Indo-Aryan''' {{Form|lang=sa|nás- ~ nā́s-|n.|nose}} + "
        "<big>-ra-</big>\n"
    )
    assert nured.source_lemma(text) == "*āč[iya]n[a]-"
    assert nured.source_heads(text) == ["nás-", "nā́s-", "-ra-"]


def test_cdial_addenda_are_canonicalized_to_reviewed_main_entries():
    index, ids, merges = nured.cdial_index(nured.DEFAULT_CDIAL, nured.DEFAULT_MERGES)
    assert merges["14349"] == "2680"
    assert {entry_id for entry_id, _ in index[nured.normalize_head("kaṇṭhá-")]} == {"2680"}
    assert {"14349", "2680"} <= ids


def test_rendered_article_is_source_linked_sanitized_and_collision_safe():
    page = {"pageid": 42, "revid": 77, "title": "*test"}
    raw = '''
      <div class="mw-parser-output"><meta property="x" />
      <div class="mw-heading"><h2 id="Commentary">Commentary</h2>
      <span class="mw-editsection">edit</span></div>
      <p onclick="bad()">See <a href="/wiki/Other">other</a>
      <a href="javascript:bad()">bad</a><sup id="cite_ref-1"><a href="#cite_note-1">1</a></sup>.</p>
      <ol><li id="cite_note-1">A note.</li></ol><!-- parser report --></div>
    '''
    result = nured.sanitize_article(raw, page)
    assert "mw-editsection" not in result
    assert "parser report" not in result
    assert "<meta" not in result
    assert 'href="https://nured.org/wiki/Other"' in result
    assert 'id="nured-42-cite_ref-1"' in result
    assert 'href="#nured-42-cite_note-1"' in result
    assert "onclick" not in result and "javascript:" not in result
    assert "oldid=77" in result
    assert "NurED article:" in result


def test_source_citation_uses_stable_page_and_revision_ids():
    page = {"pageid": 1435, "revid": 3672, "timestamp": "2026-08-03T15:08:55Z"}
    assert nured.source_citation(page) == "nured[page 1435, revision 3672, 2026-08-03]"


def test_unchanged_live_inventory_does_not_create_a_date_only_refresh(tmp_path):
    prior = {
        "snapshot_date": "2026-08-18",
        "pages_sha256": "same",
        "namespace_0_pages": 875,
        "hard_redirects_excluded": 770,
        "nonredirect_pages_audited": 105,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(prior), encoding="utf-8")
    current = {**prior, "snapshot_date": "2026-08-25"}
    rows = [{"Snapshot_Date": "2026-08-25"}]
    nured.retain_snapshot_date_when_unchanged(rows, current, path)
    assert current["snapshot_date"] == rows[0]["Snapshot_Date"] == "2026-08-18"


def test_checked_in_snapshot_is_complete_and_reproducible():
    audit = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-audit.csv")
    blocks = read_rows(ROOT / "data/other/entry_texts/20260818-nured-org.csv")
    manifest = json.loads(
        (ROOT / "data/other/forms/raw_data/20260818-nured-org-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(audit) == manifest["nonredirect_pages_audited"]
    assert len({row["Entry_Key"] for row in audit}) == len(audit)
    assert len(blocks) == manifest["installed_text_blocks"]
    assert Counter(row["Status"] for row in audit) == manifest["audit_status_counts"]
    assert Counter(row["Article_Type"] for row in audit if row["Article_Type"]) == {
        key: value for key, value in manifest["article_type_counts"].items()
    }
    assert sum(manifest["article_type_counts"].values()) == manifest["scoped_articles"]
    assert {
        row["Page_ID"] for row in audit if row["Status"] == "unmatched"
    } >= {"909", "1352"}
    assert all(len(row["Wikitext_SHA256"]) == 64 for row in audit)
    assert all("�" not in row["Raw_Wikitext"] + row["Rendered_HTML"] for row in audit)
    assert all(row["Source"].startswith("nured[page ") for row in blocks)
    assert all(row["Format"] == "html" and row["Kind"] == "etymology" for row in blocks)
    assert len({(row["Form_ID"], row["Position"]) for row in blocks}) == len(blocks)
    assert all("NurED article:" in row["Content"] for row in blocks)


def test_seeded_twenty_article_review_sample_has_no_material_errors():
    audit = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-audit.csv")
    scoped = [row for row in audit if row["Article_Type"]]
    expected = {
        row["Page_ID"] for row in random.Random(20260818).sample(scoped, 20)
    }
    sample = read_rows(ROOT / "data/other/forms/raw_data/20260818-nured-org-sample.csv")
    assert {row["Page_ID"] for row in sample} == expected
    assert all(row["Raw_Compared"] == row["Rendered_Compared"] == "yes" for row in sample)
    assert all(row["Material_Error"] == "no" for row in sample)


def test_all_nured_blocks_survive_the_compiled_sidecar():
    raw = read_rows(ROOT / "data/other/entry_texts/20260818-nured-org.csv")
    compiled = read_rows(ROOT / "cldf/entry-texts.csv")
    aliases = {
        row["Legacy_ID"]: row["Form_ID"]
        for row in read_rows(ROOT / "cldf/form-id-aliases.csv")
    }
    compiled_keys = {
        (row["Form_ID"], row["Position"], row["Source"])
        for row in compiled
    }
    for row in raw:
        key = (aliases.get(row["Form_ID"], row["Form_ID"]), row["Position"], row["Source"])
        assert key in compiled_keys
