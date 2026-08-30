"""Focused guards for the source-local Koch post-freeze package."""

import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_kochbd_2011_manual"
SCRIPT = PACKAGE / "build_post_freeze_package.py"
MANUAL_MANIFEST = PACKAGE / "source_manifest.json"
POST_FREEZE_MANIFEST = PACKAGE / "post_freeze_manifest.json"
RECONCILIATION = PACKAGE / "reconciliation.tsv"
STAGING_AUDIT = PACKAGE / "staging_audit.tsv"
STAGED_FORMS = PACKAGE / "staged_forms.csv"
SITE_METADATA = PACKAGE / "site_metadata.tsv"
REFERENCE_METADATA = PACKAGE / "reference_metadata.json"
EXCLUSION_POLICY = PACKAGE / "exclusion_policy.json"
SOUND_INVENTORY = PACKAGE / "sound_inventory.tsv"
SOUND_PROFILE = PACKAGE / "sound_profile.txt"
SOUND_DECISIONS = PACKAGE / "sound_profile_decisions.json"
SHARED_INTEGRATION_MANIFEST = PACKAGE / "shared_integration_manifest.json"
INSTALLED_FORMS = ROOT / "data/other/forms/20260826-sil-kochbd.csv"
SHARED_PROFILE = ROOT / "conversion/sil-bangladesh.txt"
DIALECT_REGISTRY = ROOT / "cldf/dialects.csv"
SOURCES_BIB = ROOT / "cldf/sources.bib"
BUILD_SCRIPT = ROOT / "make_cldf.py"
SOURCE_CHECKLIST = ROOT / "source_checklists/20260826-sil-kochbd.md"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_post_freeze_generator_is_exact_and_reproducible():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    first_manifest = POST_FREEZE_MANIFEST.read_bytes()
    first_outputs = {
        path.name: sha256(path)
        for path in (
            RECONCILIATION, STAGING_AUDIT, STAGED_FORMS, SITE_METADATA,
            REFERENCE_METADATA, EXCLUSION_POLICY, SOUND_INVENTORY,
            SOUND_PROFILE, SOUND_DECISIONS,
        )
    }
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    assert POST_FREEZE_MANIFEST.read_bytes() == first_manifest

    manifest = json.loads(POST_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "source_local_post_freeze_complete"
    assert manifest["manual_manifest_sha256"] == sha256(MANUAL_MANIFEST)
    assert manifest["source_pdf_sha256"] == (
        "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"
    )
    assert manifest["manual_conceptual_cells"] == 2149
    assert manifest["manual_expanded_rows"] == 2159
    assert manifest["legacy_expanded_rows"] == 2208
    assert manifest["staged_rows"] == 1017
    assert manifest["excluded_rows"] == 1142
    assert manifest["outputs"] == first_outputs
    assert manifest["mixed_resolved_unresolved_coordinates"] == ["item-241/site-r"]
    assert manifest["ambiguity_only_conceptual_cells"] == 225
    assert manifest["coordinates_with_unresolved_variants"] == 226
    assert manifest["unresolved_expanded_rows"] == 226
    assert len(manifest["unresolved_lexical_coordinates"]) == 226
    assert len(manifest["deferred_shared_actions"]) == 8


def test_reconciliation_is_exhaustive_and_preserves_legacy_key_identity():
    rows = read_tsv(RECONCILIATION)
    mapped = [row for row in rows if row["manual_entry_key"]]
    assert len(rows) == 2208
    assert len(mapped) == 2187
    assert len({row["manual_entry_key"] for row in mapped}) == 2159
    assert Counter(row["comparison"] for row in rows) == {
        "form_exact": 705,
        "form_difference": 568,
        "manual_recovered_legacy_unresolved": 540,
        "manual_excludes_legacy_installed_ambiguous": 207,
        "manual_ambiguous_legacy_unresolved": 23,
        "blank_match": 25,
        "not_used_match": 119,
        "legacy_spurious_not_used_collision": 21,
    }
    assert sum(row["legacy_alias_retired"] == "true" for row in rows) == 25
    spurious = [
        row for row in rows
        if row["comparison"] == "legacy_spurious_not_used_collision"
    ]
    assert {row["item"] for row in spurious} == {"7", "10", "12"}
    assert all(not row["manual_entry_key"] for row in spurious)

    item_241_r = [
        row for row in read_tsv(STAGING_AUDIT)
        if row["item"] == "241" and row["site_code"] == "r"
    ]
    assert [(row["entry_key"], row["status"], row["form"], row["visible_base"])
            for row in item_241_r] == [
        ("silkochbd2011:i241:r:2", "attested", "akui̯ʃa", ""),
        ("silkochbd2011:i241:r:1", "ambiguous", "", "tɛp"),
    ]


def test_staging_and_exclusions_are_complete_and_disjoint():
    with STAGED_FORMS.open(encoding="utf-8", newline="") as stream:
        staged = list(csv.reader(stream))
    audit = read_tsv(STAGING_AUDIT)
    exclusions = json.loads(EXCLUSION_POLICY.read_text(encoding="utf-8"))

    assert len(staged) == 1017
    assert {len(row) for row in staged} == {15}
    assert len({row[10] for row in staged}) == 1017
    assert all(row[0] == "Koch" and row[2] and row[5] == row[2] for row in staged)
    assert all(row[10].split(":")[2] in {"b", "c", "q", "r"} for row in staged)
    assert Counter(row["disposition"] for row in audit) == {
        "staged_target": 1017,
        "excluded_control": 772,
        "excluded_ambiguous": 226,
        "excluded_blank": 25,
        "excluded_not_used": 119,
    }
    assert len(audit) == 2159
    assert exclusions["source_expanded_rows"] == 2159
    assert exclusions["staged_rows"] == 1017
    assert exclusions["excluded_rows"] == 1142
    assert "No unresolved modifier is inferred" in exclusions["ambiguity_policy"]
    assert "No cognate or borrowing edge" in exclusions["etymology_policy"]


def test_source_local_site_and_reference_metadata_are_exact():
    sites = read_tsv(SITE_METADATA)
    reference = json.loads(REFERENCE_METADATA.read_text(encoding="utf-8"))
    assert {row["site_code"]: (row["site_name"], row["source_variety"], row["role"])
            for row in sites} == {
        "b": ("Nokshi", "Tintekiya Koch", "target"),
        "c": ("Kholchanda", "Tintekiya Koch", "target"),
        "q": ("Uttor Nokshi", "Chapra Koch", "target"),
        "r": ("Chandabhoi", "Tintekiya Koch", "target"),
        "l": ("Bharatpur", "A’tong", "control"),
        "m": ("Nalchapra", "A’tong", "control"),
        "0": ("Bangla", "Standard Bangla", "control"),
    }
    assert all(row["latitude"] == row["longitude"] == "" for row in sites)
    assert all("no exact site coordinate" in row["coordinate_note"] for row in sites)
    assert "Bhoratpur/Bharatpur" in next(row["location"] for row in sites if row["site_code"] == "l")
    assert "Namchapra/Nolchapra/Nalchapra" in next(
        row["location"] for row in sites if row["site_code"] == "m"
    )
    assert reference["id"] == "kim-ahmad-kim-sangma2011kochbd"
    assert reference["authors"] == ["Seung Kim", "Sayed Ahmad", "Amy Kim", "Mridul Sangma"]
    assert reference["number"] == "2011-023"
    assert reference["source_pdf_pages"] == 91
    assert reference["source_pdf_sha256"] == (
        "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"
    )


def test_preservation_profile_covers_every_staged_form():
    with STAGED_FORMS.open(encoding="utf-8", newline="") as stream:
        staged = list(csv.reader(stream))
    inventory = read_tsv(SOUND_INVENTORY)
    decisions = json.loads(SOUND_DECISIONS.read_text(encoding="utf-8"))
    tokenizer = Tokenizer(str(SOUND_PROFILE))
    for row in staged:
        converted = tokenizer(row[2], column="IPA", segment_separator="", separator="")
        assert "�" not in converted
    assert len(inventory) == 44
    assert len({row["codepoint"] for row in inventory}) == 44
    assert decisions["inventory_scope"] == {
        "all_attested_rows": 1789,
        "staged_target_rows": 1017,
        "unique_codepoints": 44,
    }
    assert decisions["unresolved_mappings"] == []
    assert decisions["source_local_profile_is_exact_base_snapshot"] is True


def test_shared_install_is_the_frozen_target_stage_byte_for_byte():
    assert INSTALLED_FORMS.read_bytes() == STAGED_FORMS.read_bytes()
    assert sha256(INSTALLED_FORMS) == (
        "75a756be9c5b36d3538a3bd936232f7177623cd9067b2b18d5f3f7f2b83923fc"
    )
    with INSTALLED_FORMS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1017
    assert {len(row) for row in rows} == {15}
    assert len({row[10] for row in rows}) == 1017
    assert all(row[0] == "Koch" and row[2] == row[5] for row in rows)
    assert {row[10].split(":")[2] for row in rows} == {"b", "c", "q", "r"}


def test_shared_site_registry_matches_source_metadata_without_invented_points():
    sites = {row["site_id"]: row for row in read_tsv(SITE_METADATA)}
    with DIALECT_REGISTRY.open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    assert len(sites) == 7
    for site in sites.values():
        registered = dialects[site["site_id"]]
        assert registered["Tag"] == site["dialect_tag"]
        assert registered["Language_ID"] == site["language_id"]
        assert registered["Source_Language_ID"] == site["site_id"]
        assert registered["Name"] == site["site_name"]
        assert registered["Glottocode"] == site["glottocode"]
        assert registered["Latitude"] == registered["Longitude"] == ""
        assert registered["Quality"] == ""
        assert registered["Location"] == site["location"]


def test_shared_reference_records_exact_manual_only_provenance():
    text = SOURCES_BIB.read_text(encoding="utf-8")
    start = text.index("@techreport{kim-ahmad-kim-sangma2011kochbd,")
    end = text.index("\n}\n", start) + 3
    entry = text[start:end]
    assert "Kim, Seung and Ahmad, Sayed and Kim, Amy and Sangma, Mridul" in entry
    assert "month        = {March}" in entry
    assert "items 1--307 at the four Koch target sites b, c, q and r" in entry
    assert "225 ambiguity-only conceptual cells" in entry
    assert "none supplied or verified a reading" in entry
    assert "Aryaman Arora and OpenAI Codex" in entry
    assert "glyph table verified word by word" not in entry


def test_shared_profile_is_exact_and_parser_route_is_explicit():
    assert SHARED_PROFILE.read_bytes() == SOUND_PROFILE.read_bytes()
    assert sha256(SHARED_PROFILE) == (
        "bcaf9bcb1098d3dfe394aa2cb0003873c31417e2a50f34643acbbe9a1a349936"
    )
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    route = 'if source_key == "kim-ahmad-kim-sangma2011kochbd":'
    assert route in build
    route_block = build[build.index(route):build.index(route) + 180]
    assert 'row_ipa = "sil-bangladesh"' in route_block
    assert "row_convert = True" in route_block

    sys.path.insert(0, str(ROOT))
    from make_cldf import parse_file

    errors = io.StringIO()
    rows, stats = parse_file(
        str(INSTALLED_FORMS), errors=errors, file_num=0, param_counter={}
    )
    assert len(rows) == 1017
    assert stats == {"converted": 1017, "for_conversion": 1017}
    assert errors.getvalue() == ""
    assert all(row.old_form == row.ipa and "�" not in row.form for row in rows)


def test_shared_integration_manifest_freezes_scope_and_deferred_gates():
    manifest = json.loads(SHARED_INTEGRATION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "shared_source_specific_integration_complete"
    assert manifest["installed"] == {
        "path": "data/other/forms/20260826-sil-kochbd.csv",
        "rows": 1017,
        "sha256": sha256(INSTALLED_FORMS),
        "entry_key_policy": (
            "Immutable silkochbd2011:item:site:variant source keys; 1017 unique target "
            "keys. Matching legacy keys remain attached to the same source occurrence."
        ),
    }
    assert manifest["audit"]["dispositions"] == {
        "staged_target": 1017,
        "excluded_control": 772,
        "excluded_ambiguous": 226,
        "excluded_blank": 25,
        "excluded_not_used": 119,
    }
    assert manifest["audit"]["ambiguity_only_conceptual_cells"] == 225
    assert manifest["audit"]["coordinates_with_unresolved_variants"] == 226
    assert manifest["audit"]["unresolved_expanded_rows"] == 226
    assert manifest["audit"]["mixed_resolved_unresolved_coordinates"] == [
        "item-241/site-r"
    ]
    assert manifest["profile"]["additions"] == []
    assert manifest["profile"]["unresolved_mappings"] == []
    assert manifest["registry"]["target_site_rows"] == 4
    assert manifest["registry"]["audit_only_control_rows"] == 3
    assert len(manifest["deferred_gates"]) == 6
    checklist = SOURCE_CHECKLIST.read_text(encoding="utf-8")
    assert "Installed rows: 1017" in checklist
    assert "global source-audit regeneration" in checklist
    assert "consolidated build" in checklist
