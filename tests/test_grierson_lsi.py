import csv
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
        "Grierson1928,,,,MALAYALAM\n",
        encoding="utf-8",
    )
    return cldf


def test_import_rows_preserves_source_identity_and_page(tmp_path):
    rows = list(MODULE.import_rows(fixture_cldf(tmp_path)))
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


def test_import_languages_inherits_known_clade_and_falls_back(tmp_path):
    cldf = fixture_cldf(tmp_path)
    registry = tmp_path / "languages.csv"
    registry.write_text(
        "ID,Name,Glottocode,Latitude,Longitude,Clade,Location,Quality\n"
        "Mal,Malayalam,mala1464,9.0,76.0,S. Dravidian I,,A\n",
        encoding="utf-8",
    )
    rows = list(MODULE.import_languages(cldf, registry))
    assert rows[0][0] == "LSI-MALAYALAM"
    assert rows[0][1] == "LSI — Malayāḷam"
    assert rows[0][5] == "S. Dravidian I"
    assert rows[0][6].startswith("Lexibank LSI v1.0 coordinate")
    assert rows[1][1] == "LSI — Missing of Nowhere"
    assert rows[1][5] == "Other"


def test_registry_update_replaces_only_lsi_slice(tmp_path):
    registry = tmp_path / "languages.csv"
    registry.write_text(
        "ID,Name,Glottocode,Latitude,Longitude,Clade,Location,Quality\n"
        "Keep,Keep,keep1234,,,Other,,C\n"
        "LSI-OLD,Old,,,,Other,,B\n",
        encoding="utf-8",
    )
    count = MODULE.update_language_registry(
        registry,
        [["LSI-NEW", "New", "", "", "", "Other", "", "B"]],
    )
    assert count == 1
    with registry.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["ID"] for row in rows] == ["Keep", "LSI-NEW"]
