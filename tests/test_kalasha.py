import csv
import importlib.util
import sys
from pathlib import Path

from segments import Tokenizer

DATA_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(DATA_DIR))
from utils import mapping  # noqa: E402


SCRIPT = DATA_DIR / "data/other/forms/raw_data/kalasha.py"
PROFILE = DATA_DIR / "conversion/kalasha.txt"
SPEC = importlib.util.spec_from_file_location("kalasha_extractor", SCRIPT)
kalasha = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kalasha
SPEC.loader.exec_module(kalasha)


def test_legacy_sil_headword_decoding():
    assert kalasha.decode_legacy("a5stru") == "ástru"
    assert kalasha.decode_legacy("a}i") == "aši"
    assert kalasha.decode_legacy("a}raphŒ‡") == "ašraphí"
    assert kalasha.decode_legacy("aST") == "aṣṭ"
    assert kalasha.decode_legacy("a43%") == "ạ̃́"
    assert kalasha.decode_legacy("c4hir") == "čhir"
    assert kalasha.decode_legacy("Jac4") == "j̣ač"
    assert kalasha.decode_legacy("ch4et") == "čhet"


def test_legacy_sound_values_and_displaced_vowel_marks():
    assert kalasha.decode_legacy("aKgu5{i") == "aŋgúži"
    assert kalasha.decode_legacy("atik5") == "atík"
    assert kalasha.decode_legacy("citroy54ak") == "citrọ́yak"
    assert kalasha.decode_legacy("h|a54") == "hrạ́"


def test_kalasha_sound_profile_and_filename_routing():
    tokenizer = Tokenizer(PROFILE)
    converted = tokenizer("aŋgúži bačhọ́a j̣ač", column="IPA")
    assert converted.replace(" ", "").replace("#", " ") == "aŋgúži baʦ̣ʰọ́a ʣ̣aʦ̣"
    assert mapping["kalasha"] == "kalasha"


def test_cdial_reference_pattern_rejects_uncertain_number():
    text = "Etym: aKkusa- ‘hook’ T-111?. Etym: astru- ‘tear’ T-919."
    assert kalasha.TURNER_REFERENCE.findall(text) == ["919"]
    assert kalasha.UNCERTAIN_TURNER_REFERENCE.findall(text) == ["111"]


def test_finish_keeps_linked_unlinked_and_uncertain_entries():
    linked = kalasha.Entry("ástru", "N", 41, 11, gloss=["Tears"])
    linked.text = ["Etym:", "asru-", "T-919."]
    rows = []
    kalasha._finish(linked, rows, {"919"})
    assert rows[0][:8] == [
        "bumb", "919", "ástru", "Tears", "", "",
        "",
        "trail-cooper1999[p. 41 (printed p. 11)]",
    ]
    assert len(rows[0]) == 15

    uncertain = kalasha.Entry("aŋgúži", "N", 38, 8, gloss=["Spatula"])
    uncertain.text = ["Etym:", "ankusa-", "T-111?."]
    kalasha._finish(uncertain, rows, {"111"})
    assert rows[-1][1] == "111"
    assert rows[-1][6] == ""
    assert "uncertain" in rows[-1][14]

    unlinked = kalasha.Entry("ajáp", "Adj", 31, 1, gloss=["Remarkable"])
    kalasha._finish(unlinked, rows, {"111", "919"})
    assert rows[-1][1] == ""
    assert not rows[-1][6].startswith("uncertain;")


def test_structured_morphology_causative_loan_and_etymology():
    base = kalasha.Entry("dik", "V", 100, 70, key="e1")
    base.text = [
        "Caus: dek. Morph: di-k. From: Khowar. "
        "Etym: *dadāti ‘gives’ T-6152. Prdm: Class 1 (kárik)."
    ]
    causative = kalasha.Entry("dek", "V", 101, 71, key="e2")
    causative.text = []
    metadata = kalasha._metadata([base, causative])

    assert metadata["e1"]["borrowed_from_key"] == "tc-loan-Khowar"
    assert metadata["e1"]["etymology"].startswith("Trail & Cooper Etym: *dadāti")
    assert "Kalasha-class-1" in metadata["e1"]["tags"]
    assert metadata["e2"]["parents"] == ["e1"]
    assert "causative" in metadata["e2"]["tags"]


def test_explicit_dialect_variants_get_language_and_parent_keys():
    entry = kalasha.Entry("dur", "N", 200, 170, key="house", gloss=["House"])
    entry.text = ["Variant: han (Birir); grom (Urtsun)."]
    rows = kalasha._dialect_variants(entry, set(), {})

    assert [(row[0], row[2], row[11]) for row in rows] == [
        ("bir", "han", "house"),
        ("urt", "grom", "house"),
    ]
    assert all("dialect-variant" in row[14] for row in rows)


def test_posless_cross_reference_is_an_entry():
    line = [
        {"x0": 108.0, "x1": 137.0, "fontname": "X+SILDoulosNPBold", "size": 10,
         "text": "ane5na"},
        {"x0": 141.0, "x1": 170.0, "fontname": "X+RamnaKLS", "size": 14,
         "text": "urdu"},
        {"x0": 174.0, "x1": 188.0, "fontname": "X+TimesNewRoman,Italic", "size": 10,
         "text": "See"},
    ]
    assert kalasha._entry_start(line, 108) == ("anéna", "", 1)


def test_enriched_kalasha_cldf_graph_is_resolved():
    with (DATA_DIR / "cldf/forms.csv").open(encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [
        row for row in all_rows
        if any(source.strip().startswith("trail-cooper1999") for source in row["Source"].split(";"))
    ]
    by_id = {row["ID"]: row for row in all_rows}

    assert sum("dialect-variant" in row["Tags"].split() for row in rows) == 146
    assert {"bumb", "rumb", "bir", "urt"} <= {row["Language_ID"] for row in rows}
    assert sum(row["Relation"] == "borrowed" for row in rows) == 956
    assert sum("loan-source" in row["Tags"].split() for row in rows) == 9
    assert sum(bool(row["Etymology"]) for row in rows) == 878
    assert sum("morphology" in row["Tags"].split() for row in rows) == 173
    assert sum("causative" in row["Tags"].split() for row in rows) == 34
    assert all(not row["Origin_ID"] or row["Origin_ID"] in by_id for row in rows)
    assert all(not row["Variant_Of"] or row["Variant_Of"] in by_id for row in rows)
    assert all(not row["Borrowed_From"] or row["Borrowed_From"] in by_id for row in rows)

    with (DATA_DIR / "cldf/derivation.csv").open(encoding="utf-8") as stream:
        edges = list(csv.DictReader(stream))
    source_ids = {row["ID"] for row in rows}
    source_edges = [
        edge for edge in edges
        if edge["Child_ID"] in source_ids or edge["Parent_ID"] in source_ids
    ]
    assert len(source_edges) == 239
    assert all(edge["Child_ID"] in by_id and edge["Parent_ID"] in by_id for edge in source_edges)
