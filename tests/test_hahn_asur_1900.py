"""Source-stage checks for Hahn's complete printed 1900 Asur article."""

import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/hahn_asur_1900"
CSV = DATA / "data/other/forms/20260925-hahn-asur.csv"
PROFILE = DATA / "conversion/hahn-asur-1900.txt"
spec = importlib.util.spec_from_file_location("hahn_asur", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_whole_installation_and_historical_evidence():
    rows,audit=source.build()
    assert rows==installed() and len(rows)==835 and len(audit)==670
    historical=list(csv.reader((PACKAGE/'historical-lexical.csv').open()))
    assert len(historical)==621 and {r[10] for r in historical}<={r[10] for r in rows}
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    assert manifest['installed_forms']==835 and manifest['append_order']==98
    assert manifest['same_source_reuse']==3 and manifest['source_stage_status']=='complete_source_stage'
    assert 'public domain' in manifest['rights'].lower()
    assert (PACKAGE/'audit.jsonl').read_bytes()==(PACKAGE/'proposal-audit.jsonl').read_bytes()


def test_source_dialect_profile_and_scoped_parser():
    rows = installed()[:29]
    assert all(len(r) == 15 and r[0] == "Asuri" and r[14] == source.DIALECT for r in rows)
    assert all(r[7].startswith("hahn1900asur[p. 170, Asur–Mundari comparison") for r in rows)
    assert len({r[10] for r in rows}) == 29
    assert [r[2:4] for r in rows if r[2] == "bitil"] == [["bitil", "sand"]]
    assert all("Mundari comparison control:" in r[9] for r in rows)
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    dialect = dialects["hahn-1900-asur-dukma"]
    assert dialect[1:5] == [source.DIALECT, "Asuri", "hahn1900asur:Asur Dukmā", "Asur Dukmā (Hahn 1900)"]
    assert dialect[5:8] == ["", "", ""]
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@article{hahn1900asur,") == 1
    assert "20260925-hahn-asur.csv" in bib
    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("hātom", column="IPA").replace(" ", "") == "hātom"
    assert tokenizer("tihiŋ", column="IPA").replace(" ", "") == "tihiŋ"
    assert tokenizer("pēā", column="IPA").replace(" ", "") == "pēā"
    import make_cldf

    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == len(installed())


def test_full_article_coverage_and_grammar_preservation():
    rows, audit = source.build()
    assert rows == installed()
    assert len(rows) == 835 and len(audit) == 670
    assert len({r[10] for r in rows}) == len(rows)
    assert {int(a['printed_page']) for a in audit} == set(range(149,173))
    assert all(len(r) == 15 and r[0] == "Asuri" and source.DIALECT in r[14].split() for r in rows)
    assert all(not r[1] and not r[8] and not any(r[12:14]) for r in rows)
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    by_key = {r[10]: r for r in rows}
    assert by_key["hahn1900asur:p160:s18:01"][2:4] == ["iŋā āpuiŋ", "my father"]
    assert "poss" in by_key["hahn1900asur:p160:s18:01"][14].split()
    assert by_key["hahn1900asur:p153:s3:09"][2] == "īpil"
    assert by_key["hahn1900asur:p169:s46:08"][2] == "nīho"
    assert by_key["hahn1900asur:p168:s43:01"][2] == "mīad"
    assert by_key["hahn1900asur:p166:s36:07"][2] == "hukāyēme"
    assert by_key["hahn1900asur:p162:s25:05:variant:2"][2] == "rūlidilāŋ"
    assert by_key["hahn1900asur:p166:s35-lex:04"][2] == "rúar"
    assert "uncertain" in by_key["hahn1900asur:p166:s35-lex:04"][14].split()
    assert by_key["hahn1900asur:p164:s30:01:variant:2"][2] == "rāēkāiŋ"
    assert by_key["hahn1900asur:p153:s3:06"][2] == "hoṛ"
    assert by_key["hahn1900asur:p161:s19-cont:05"][2] == "seneā"
    assert by_key["hahn1900asur:p171:s49:05"][2] == "eŋā"
    assert not any(r[2] == "iyyō" for r in rows)
    assert by_key["hahn1900asur:p171:s50-right:02"][2] == "aṉyāṉ"
    tokenizer = Tokenizer(str(PROFILE))
    assert all("�" not in tokenizer(r[2], column="IPA") for r in rows)
    assert tokenizer("aṉyāṉ", column="IPA").replace(" ", "") == "aṉyāṉ"
