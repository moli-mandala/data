import csv
import io
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/grierson_lsi.py"
SPEC = importlib.util.spec_from_file_location("grierson_lsi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixture_cldf(tmp_path: Path) -> Path:
    cldf = tmp_path / "cldf"
    cldf.mkdir()
    (cldf / "parameters.csv").write_text(
        "ID,Name,Concepticon_ID,Concepticon_Gloss,DSAL_URL,PageNumber,Scans\n"
        "1_one,One,1493,ONE,https://example.test,2-3,044 045\n",
        encoding="utf-8",
    )
    (cldf / "languages.csv").write_text(
        "ID,Name,Glottocode,Glottolog_Name,ISO639P3code,Macroarea,Latitude,Longitude,"
        "Family,NameInSource,NumberInSource,Order,FamilyInSource,SubGroup\n"
        "MALAYALAM,MALAYALAM,mala1464,Malayalam,mal,Eurasia,9.59,76.52,Dravidian,"
        "Malayāḷam,,1,Dravidian,\n"
        "MISSING,MISSING,,Missing,zzz,,1.0,2.0,,of Nowhere,,2,,\n",
        encoding="utf-8",
    )
    (cldf / "forms.csv").write_text(
        "ID,Local_ID,Language_ID,Parameter_ID,Value,Form,Segments,Comment,Source,"
        "Cognacy,Loan,Graphemes,Profile\n"
        "MALAYALAM-1_one-1,,MALAYALAM,1_one,onnu,onnu,o n̪ n u,printed note,"
        "Grierson1928,,,,MALAYALAM\n"
        "MISSING-1_one-1,,MISSING,1_one,unknown,unknown,u n k n o w n,,"
        "Grierson1928,,,,MISSING\n",
        encoding="utf-8",
    )
    return cldf


def fixture_registry(tmp_path: Path) -> Path:
    registry = tmp_path / "languages.csv"
    registry.write_text(
        "ID,Name,Glottocode,Latitude,Longitude,Clade,Location,Quality\n"
        "Mal,Malayalam,mala1464,9.0,76.0,S. Dravidian I,,A\n"
        "LSI-OLD,Old,,,,Other,,B\n",
        encoding="utf-8",
    )
    return registry


def test_parent_resolution_uses_existing_language_and_skips_unmatched(tmp_path):
    parents, decisions = MODULE.resolve_parents(
        fixture_cldf(tmp_path), fixture_registry(tmp_path)
    )
    assert parents["MALAYALAM"]["ID"] == "Mal"
    assert "MISSING" not in parents
    assert decisions["MALAYALAM"] == "unique existing-language name"
    assert decisions["MISSING"] == "no existing parent"


def test_historical_name_outranks_conflicting_modern_glottocode(tmp_path):
    cldf = fixture_cldf(tmp_path)
    with (cldf / "languages.csv").open("a", encoding="utf-8") as stream:
        stream.write(
            "CONFLICT,CONFLICT,modern1234,Modern Name,zzz,Eurasia,1,2,Family,"
            "Historical Name,,3,,\n"
        )
    registry = fixture_registry(tmp_path)
    with registry.open("a", encoding="utf-8") as stream:
        stream.write("Historical,Historical Name,other1234,,,Other,,C\n")
        stream.write("Modern,Modern Name,modern1234,,,Other,,C\n")
    parents, _ = MODULE.resolve_parents(cldf, registry)
    assert parents["CONFLICT"]["ID"] == "Historical"


def test_import_rows_preserves_alias_identity_and_page(tmp_path):
    cldf = fixture_cldf(tmp_path)
    parents, _ = MODULE.resolve_parents(cldf, fixture_registry(tmp_path))
    rows = list(MODULE.import_rows(cldf, parents))
    assert rows == [[
        "LSI-MALAYALAM",
        "",
        "onnu",
        "One",
        "",
        "on̪nu",
        "printed note",
        "grierson-lsi1928[p. 2-3, form MALAYALAM-1_one-1, concept 1_one]",
        "",
        "",
        "grierson-lsi1928:MALAYALAM-1_one-1",
        "",
        "",
        "",
        "",
    ]]


def test_build_converts_clts_ipa_and_preserves_grierson_form(tmp_path):
    from make_cldf import parse_file

    source = tmp_path / "lsi.csv"
    source.write_text(
        "LSI-KHETRANI,x,ch’ī,Six,,tʃʰiː,,"
        "grierson-lsi1928[p. 12-13],,,key,,,,\n",
        encoding="utf-8",
    )
    rows, stats = parse_file(
        str(source), io.StringIO(), name="grierson"
    )
    assert stats == {"converted": 1, "for_conversion": 1}
    assert len(rows) == 1
    assert rows[0].form == "cʰī"
    assert rows[0].ipa == "tʃʰiː"
    assert rows[0].old_form == "ch’ī"


def test_import_dialects_attaches_source_lect_to_parent(tmp_path):
    cldf = fixture_cldf(tmp_path)
    parents, _ = MODULE.resolve_parents(cldf, fixture_registry(tmp_path))
    rows = list(MODULE.import_dialects(cldf, parents))
    assert len(rows) == 1
    assert rows[0][0] == "lsi_malayalam"
    assert rows[0][2:5] == [
        "Mal", "LSI-MALAYALAM", "Malayāḷam (LSI 1928)"
    ]
    assert rows[0][8] == "S. Dravidian I"
    assert rows[0][9].startswith("Lexibank LSI v1.0 coordinate")


def test_registry_updates_remove_languages_and_replace_dialect_slice(tmp_path):
    registry = fixture_registry(tmp_path)
    assert MODULE.remove_legacy_languages(registry) == 1
    with registry.open(encoding="utf-8", newline="") as stream:
        assert [row["ID"] for row in csv.DictReader(stream)] == ["Mal"]

    dialects = tmp_path / "dialects.csv"
    dialects.write_text(
        "ID,Tag,Language_ID,Source_Language_ID,Name,Glottocode,Latitude,Longitude,"
        "Clade,Location,Quality\n"
        "keep,dialect:Mal:keep:Keep,Mal,keep,Keep,mala1464,,,S. Dravidian I,,C\n"
        "old,dialect:Mal:LSI-OLD:Old,Mal,LSI-OLD,Old,mala1464,,,S. Dravidian I,,B\n",
        encoding="utf-8",
    )
    count = MODULE.update_dialect_registry(
        dialects,
        [[
            "new", "dialect:Mal:LSI-NEW:New", "Mal", "LSI-NEW", "New",
            "mala1464", "", "", "S. Dravidian I", "", "B",
        ]],
    )
    assert count == 1
    with dialects.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["ID"] for row in rows] == ["keep", "new"]
