"""Whole-source proposal contracts; no database or canonical writes."""
import csv
import importlib.util
import io
import json
import sys
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest
from segments.tokenizer import Tokenizer

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA / "data/other/forms/raw_data/dalton_haldar_juang_1872"
OUT = ROOT / "whole-source-recovery"
sys.path.insert(0, str(DATA))


@pytest.fixture(scope="module")
def proposal():
    spec = importlib.util.spec_from_file_location("juang_whole", ROOT / "prepare_whole_source.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def keyed(proposal):
    return {r[10].removeprefix("dalton1872haldarjuang:"): r for r in proposal[0]}


def test_whole_cell_coverage_and_legacy_identity(proposal):
    rows, audit, old_keys = proposal
    historical = [a for a in audit if "historical_decision" in a]
    assert len(historical) == len({a["source_unit_key"] for a in historical}) == 361
    assert sum(a["status"] == "printed-blank" for a in historical) == 153
    recovered = [a for a in historical if a["historical_decision"]["status"] == "held"]
    assert len(recovered) == 19 and all(a["entry_keys"] for a in recovered)
    assert len(rows) == len({r[10] for r in rows}) == 226
    assert len(old_keys) == 196 and old_keys <= {r[10] for r in rows}
    with (ROOT / "20260925-dalton-haldar-juang.csv").open() as handle:
        historical_order = [r[10] for r in csv.reader(handle)]
    assert [r[10] for r in rows[:196]] == historical_order
    assert all(not a["entry_keys"] for a in historical if a["status"] == "printed-blank")


def test_exact_reconciled_literals_and_distinct_witnesses(proposal):
    at = keyed(proposal)
    expected = {
        "p236:row:01":"ijing", "p236:row:19:v2":"mánush", "p236:row:23:v1":"kano",
        "p236:row:23:v2":"kan-da", "p236:row:20":"juangle", "p236:row:33":"lálái",
        "p236:row:37":"orai", "p236:row:32":"kujindá", "p237:row:14":"hassen-yá",
        "p237:row:16":"onúku", "p235:row:04":"gandá-ni", "p241:numerals:04":"ganda mi",
        "p241:numerals:01":"moiñ", "p241:miscellaneous-words:13":"orui", "p242:left-column:16":"bá", "p242:left-column:18":"ká",
        "p242:left-column:46":"ka", "p242:left-column:47":"ka loko",
        "p242:right-column:09":"níkijiko", "p242:right-column:40":"kakah", "p242:right-column:41":"kakou",
    }
    assert {k:at[k][2] for k in expected} == expected
    assert [at[f"p242:left-column:{n:02}"][2] for n in (38,39,40)] == ["niso"] * 3
    assert at["p236:row:08"][2] == "bokú" and at["p242:left-column:09"][2] == "boká"


def test_source_qualifiers_are_notes_not_etymology(proposal):
    at = keyed(proposal)
    for k in ("p236:row:21", "p236:row:22", "p237:row:14", "p237:row:16", "p237:row:17", "p237:row:20"):
        assert "question mark" in at[k][6] and "uncertain" in at[k][14].split()
        assert "?" not in at[k][2] and "(P.)" not in at[k][6]
    assert "(s.)" in at["p236:row:19:v2"][6]
    assert all(not r[9] and not r[12] and not r[13] for r in proposal[0])
    assert all(not r[4] and not r[5] for r in proposal[0])


def test_explicit_grammar_and_grouped_cases(proposal):
    at = keyed(proposal)
    for k, tags in {
        "p235:row:14":{"pron","1sg"}, "p235:row:15":{"pron","1sg","gen","poss"},
        "p239:row:03":{"verb","copula","pres","1sg"}, "p240:row:07":{"verb","pret","1sg"},
        "p242:right-column:09":{"verb"}, "p242:morphology:loko":{"pl"},
        "p242:morphology:andesh":{"m"}, "p242:morphology:petah":{"f"},
    }.items():
        assert tags <= set(at[k][14].split())
    for section in ("04","06"):
        assert not at[f"p241:pronouns:{section}:v2"][11]
        assert "without pairing" in at[f"p241:pronouns:{section}:v2"][6]
    keys = {r[10] for r in proposal[0]}
    assert all(not r[11] or r[11] in keys and r[11] != r[10] for r in proposal[0])


def test_prose_scope_and_source_attribution(proposal):
    at = keyed(proposal)
    assert at["p152:prose:tooth"][2] == "goné"
    assert at["p157:prose:priest"][2] == "Nágam"
    assert at["p156:prose:tumba"][2] == "Tumba" and "uncertain" in at["p156:prose:tumba"][14].split()
    prose = [a for a in proposal[1] if ":prose:" in a["source_unit_key"]]
    assert len(prose) == 9 and sum(bool(a["entry_keys"]) for a in prose) == 4
    assert all(not a["entry_keys"] for a in prose if a["form"] in ("Asan","Kopin","Vaitarani","chob","sikas"))
    assert all("dialect:" not in r[14] for r in proposal[0] if ":prose:" in r[10])


def test_literal_accent_profile_and_nfd_parity(proposal):
    tokenizer = Tokenizer(str(OUT / "proposal-profile.txt"))
    def norm(value):
        return unicodedata.normalize("NFC", tokenizer(value, column="IPA").replace(" ", "").replace("#", " "))
    assert norm("bhagwán") == "bʰagván"
    assert norm("Nágam") == "nágam" and norm("moiñ") == "moiñ"
    for row in proposal[0]:
        assert "�" not in norm(row[2])
        assert norm(row[2]) == norm(unicodedata.normalize("NFD", row[2]))


def test_actual_metadata_parser_preserves_original_and_tags(proposal, tmp_path):
    import make_cldf
    import source_meta
    rows = proposal[0]
    path = tmp_path / "other/forms/20260925-dalton-haldar-juang.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    meta = source_meta.SourceMeta([OUT / "20260925-dalton-haldar-juang.yaml"])
    errors = io.StringIO()
    with patch.object(source_meta, "load", return_value=meta), patch.dict(make_cldf.convertors, {
        "dalton-haldar-juang-1872":Tokenizer(str(OUT / "proposal-profile.txt"))
    }):
        parsed, stats = make_cldf.parse_file(str(path), errors, name=path.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == len(rows)
    raw = {r[10]:r for r in rows}
    assert {r.entry_key for r in parsed} == set(raw)
    for row in parsed:
        expected = raw[row.entry_key]
        assert (row.old_form, row.native, row.ipa, row.notes, row.source, row.gloss) == tuple(expected[i] for i in (2,4,5,6,7,3))
        assert set(expected[14].split()) <= set(row.tags.split())


def test_registered_language_dialect_and_citation_scope(proposal):
    from pybtex.database import parse_file
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    bib = parse_file(str(OUT / "proposal-sources.bib"))
    assert set(bib.entries) == {"dalton1872haldarjuang"}
    assert all(r[7].split("[", 1)[0] in bib.entries and ";" not in r[7] for r in proposal[0])
    languages = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    dialects = {r[1] for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    for row in proposal[0]:
        assert row[0] in languages
        for tag in row[14].split():
            assert tag in dialects or tag in GRAMMATICAL_TAGS or tag in GENDER_TAGS
