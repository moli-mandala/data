"""Integration contracts for the three frozen manual survey packages."""
from __future__ import annotations

import csv
import importlib.util
import io
import shutil
from collections import Counter
from pathlib import Path

import pytest
from pybtex.database import parse_file as parse_bib

import make_cldf
import unify_cldf
import burushaski_comparisons
from assign_form_ids import assign_ids
from dialects import load_dialect_aliases, normalize_dialect

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/other/forms/raw_data"
SPEC = importlib.util.spec_from_file_location("manual_surveys_integration", RAW / "integrate_manual_surveys_2026.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def dict_rows(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


@pytest.mark.parametrize("name", adapter.SOURCES)
def test_frozen_records_install_with_canonical_languages_and_stable_keys(name):
    spec = adapter.SOURCES[name]
    prepared, audit = adapter.prepare(name)
    adapter.validate_registry(prepared)
    with (RAW.parent / spec["output"]).open(newline="") as stream:
        installed = list(csv.reader(stream))
    assert installed == [[row[k] for k in adapter.FIELDS] for row in prepared]
    assert len(installed) == spec["count"]
    assert {r[0] for r in installed} == {spec["parent"]}
    assert len({r[10] for r in installed}) == spec["count"]
    assert dict_rows(RAW / f"20260914-sil-{name}-integration-audit.csv") == audit
    assert all(not row[1] and not row[8] and not row[9] and not row[12] and not row[13] for row in installed)


@pytest.mark.parametrize("name", adapter.SOURCES)
def test_real_compiler_preserves_all_records_and_transcription_layers(name):
    spec = adapter.SOURCES[name]
    original, _ = adapter.prepare(name)
    by_key = {r["Entry_Key"]: r for r in original}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(RAW.parent / spec["output"]), errors, file_num=spec["output"].removesuffix(".csv"))
    assert not errors.getvalue()
    assert stats == {"converted": spec["count"], "for_conversion": spec["count"]}
    assert len(parsed) == spec["count"]
    assert {r.entry_key for r in parsed} == set(by_key)
    for row in parsed:
        raw = by_key[row.entry_key]
        assert row.lang == spec["parent"]
        assert row.old_form == raw["Form"]
        assert row.ipa == raw["Phonemic"]
        assert row.source == raw["Source"]
        assert row.variant_of_key == raw["Variant_Of_Key"]
        assert row.is_lone and not row.param
        assert "�" not in row.form and row.form.strip()
        if name == "ho":
            assert row.form == raw["Form"]


def test_language_mapping_qualifiers_and_unknown_coordinates():
    dialects = {r["Tag"]: r for r in dict_rows(ROOT / "cldf/dialects.csv")}
    aliases = load_dialect_aliases(ROOT / "cldf/dialects.csv")
    for name, spec in adapter.SOURCES.items():
        rows, _ = adapter.prepare(name)
        for row in rows:
            parent, _ = normalize_dialect(row["Language_ID"], row["Tags"], aliases)
            assert parent == spec["parent"]
            d = dialects[row["Tags"].split()[0]]
            assert d["Language_ID"] == spec["parent"]
            assert not d["Latitude"] and not d["Longitude"] and not d["Glottocode"]
            assert d["Location"] and d["Quality"] == "C"
    rows, audit = adapter.prepare("bhumij")
    udala = [r for r in rows if "1989-udala" in r["Entry_Key"]]
    assert udala and all("uncertain" in r["Tags"].split() for r in udala)
    assert sum(r["Notes"] == "small (source qualifier)" for r in rows) == 2
    marked = next(r for r in rows if r["Entry_Key"] == "bhumij1989-ladhiramsai-i195-a01")
    assert marked["Notes"] == "Source qualifier: (?)"
    assert "uncertain" in marked["Tags"].split()
    assert "source-qualification:" in next(r for r in audit if r["Entry_Key"] == marked["Entry_Key"])["Uncertainty"]
    ho, audit = adapter.prepare("ho")
    corrected = [r for r in ho if r["Entry_Key"].endswith("-i093")]
    assert len(corrected) == 14 and {r["Gloss"] for r in corrected} == {"tail"}
    assert sum(bool(r["Correction"]) for r in audit) == 14


def test_seeded_frozen_ledger_to_installed_audit_has_no_material_errors():
    audit = adapter.sample_audit(20260915)
    assert all(s["sample_size"] == 20 and s["material_errors"] == 0 for s in audit["sources"].values())


def test_references_are_complete_and_authors_match_source_covers():
    bib = parse_bib(ROOT / "cldf/sources.bib")
    formatted = {r["ID"]: r for r in dict_rows(ROOT / "cldf/references.csv")}
    for spec in adapter.SOURCES.values():
        ref = bib.entries[spec["source"]]
        assert ref.fields["ocr"] == "No"
        assert ref.fields["etymology_provenance"] == "none"
        assert "Appendix" in ref.fields["included"]
        assert spec["output"] in ref.fields["provenance"]
        assert "SHA-256" in ref.fields["provenance"]
        rendered = formatted[spec["source"]]
        assert rendered["OCR"] == "No" and rendered["Etymology_Provenance"] == "none"
        assert rendered["Source"] and rendered["Editor"]
        assert "Appendix" in rendered["Progress"]
        assert spec["output"] in rendered["Provenance"]
    authors = bib.entries["josephmichael2021dhurwa"].persons["author"]
    assert [str(p) for p in authors] == ["Joseph, D. Selwyn", "Joseph, Selvi"]


def test_display_profiles_preserve_difficult_source_notation():
    def convert(profile, text):
        return make_cldf.convertors[profile](text, column="IPA").replace(" ", "").replace("#", " ")
    assert convert("sil-ho", "boʔo, bo?o ṯaḏ ɖ ẽ") == "boʔo, bo?o ṯaḏ ɖ ẽ"
    assert convert("sil-bhumij", "ɖɳʈ tʃ dʒ ʌː") == "ḍṇṭ c j ā"
    assert convert("sil-dhurwa-2021", "ʈɛl dʒ j bom:a") == "ṭel j y bomːa"


def test_new_sources_do_not_shift_old_input_positions():
    existing = sorted([
        "data/cdial/cdial.csv", "data/munda/forms.csv", "data/dedr/dedr_new.csv", "data/dedr/pdr.csv",
        *[str(p.relative_to(ROOT)) for p in (ROOT / "data/other/forms").glob("*.csv")
          if str(p.relative_to(ROOT)) != make_cldf.MERRIAM_DRAVIDIAN_DB_FILE
          and str(p.relative_to(ROOT)) not in make_cldf.APPENDED_SURVEY_FILES],
    ]) + [make_cldf.MERRIAM_DRAVIDIAN_DB_FILE, "data/dbia/forms.csv", *make_cldf.WESTERN_SURVEY_FILES]
    assert not set(existing) & set(make_cldf.MANUAL_SURVEY_FILES)
    assert make_cldf.APPENDED_SURVEY_FILES == make_cldf.WESTERN_SURVEY_FILES + make_cldf.MANUAL_SURVEY_FILES + (make_cldf.SHETH_FILE, make_cldf.SHETH_SANSKRIT_FILE)


def test_source_only_compilation_retains_homonyms_variants_and_durable_identity(tmp_path, monkeypatch):
    """Exercise real compilation/deduplication on these sources with empty other inputs.

    This is a bounded fixture, not a substitute for the full repository build.
    """
    empty_files = [
        "data/cdial/cdial.csv", "data/munda/forms.csv", "data/dedr/dedr_new.csv", "data/dedr/pdr.csv",
        make_cldf.MERRIAM_DRAVIDIAN_DB_FILE, "data/dbia/forms.csv",
        "data/cdial/params.csv", "data/munda/params.csv", "data/dedr/params.csv", "data/dbia/params.csv",
        "data/etymologies.csv", *make_cldf.WESTERN_SURVEY_FILES, make_cldf.SHETH_FILE,
    ]
    for filename in empty_files:
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    (tmp_path / "cldf").mkdir()
    for filename in ["cldf/languages.csv", "cldf/dialects.csv", *make_cldf.MANUAL_SURVEY_FILES]:
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    (tmp_path / "data/nuristani_cognates.csv").write_text("Ancestor_ID\n")
    for filename in ["data/cross-family-comparisons.csv", "data/manual-cross-family-comparisons.csv", "data/dbia/comparisons.csv"]:
        (tmp_path / filename).write_text(",".join(make_cldf.CROSS_FAMILY_COLUMNS) + "\n")
    monkeypatch.chdir(tmp_path)
    for name in ["lang_set", "param_set", "included_params"]:
        monkeypatch.setattr(make_cldf, name, set())
    make_cldf.main()
    rows = dict_rows(tmp_path / "cldf/forms.csv")
    assert len(rows) == 5809
    assert not (tmp_path / "errors.txt").read_text()
    assert Counter(r["Language_ID"] for r in rows) == Counter(ho=2900, mu=2100, Parji=809)
    assert len({r["Entry_Key"] for r in rows}) == 5809
    by_key = {r["Entry_Key"]: r for r in rows}
    variants = [r for r in rows if r["Variant_Of_Key"]]
    assert len(variants) == 46
    assert all(by_key[r["Variant_Of_Key"]]["Language_ID"] == r["Language_ID"] for r in variants)
    (tmp_path / "data/strand_oia_redirects.csv").write_text("Strand_ID,CDIAL_ID\n")
    monkeypatch.setattr(unify_cldf, "load_burushaski_catalog", lambda: [])
    monkeypatch.setattr(unify_cldf, "append_burushaski_comparisons", lambda rows: burushaski_comparisons.append_comparisons(rows, tmp_path / "cldf/comparisons.csv"))
    monkeypatch.setattr(unify_cldf, "write_burushaski_comparison_audit", lambda rows: burushaski_comparisons.write_audit(rows, tmp_path / "data/burushaski-audit.csv"))
    unify_cldf.main()
    unified = dict_rows(tmp_path / "cldf/forms.csv")
    edges = dict_rows(tmp_path / "cldf/edges.csv")
    assert len(unified) == 5809 and len(edges) == 46
    assert Counter(r["Status"] for r in unified) == Counter(unlinked=5763, **{"": 46})
    assert all(e["Kind"] == "variant" and e["Rank"] == "1" for e in edges)
    expected_edges = {(r["ID"], by_key[r["Variant_Of_Key"]]["ID"]) for r in variants}
    assert {(e["Child_ID"], e["Parent_ID"]) for e in edges} == expected_edges
    keys = {r["ID"]: r["Entry_Key"] for r in rows}
    ids, registry = assign_ids(unified, [], keys)
    corrected = [dict(r, Form=r["Form"] + "x", Original=r["Original"] + "x", Gloss=r["Gloss"] + " corrected") for r in reversed(unified)]
    reordered_ids, _ = assign_ids(corrected, registry, keys)
    assert ids == reordered_ids
