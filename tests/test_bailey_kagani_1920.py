"""Focused source-input checks for Bailey's complete Kāgānī vocabulary."""

import csv
import importlib.util
import io
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/bailey_kagani_1920"
CSV = DATA / "data/other/forms/20260925-bailey-kagani.csv"
PROFILE = DATA / "conversion/bailey-kagani-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_kagani_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_full_source_regeneration_and_keys():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 261 and len(rows) == 329
    assert Counter(a["status"] for a in audit) == {"ingested": 258, "exclude_crossref": 3}
    assert len({r[10] for r in rows}) == 329
    legacy = json.loads((PACKAGE / "legacy-entry-keys.json").read_text())
    assert len(legacy) == 83 and set(legacy) <= {r[10] for r in rows}
    assert not any("withheld" in a["printed_form_review"] for a in audit)
    assert {(a["printed_page"], a["column"], a["editorial_line_number"]) for a in audit if a["status"] == "exclude_crossref"} == {(108,"right",3),(108,"right",10),(108,"right",35)}


def test_expansion_grammar_and_rare_readings():
    rows = {r[10]:r for r in installed()}
    def cell(page, column, line, answer=1):
        key = f"bailey1920kagani:p{page}:{column}:line:{line}"
        return rows[key if answer == 1 else f"{key}:answer{answer}"]
    assert cell(106,"left",24,3)[2:4] == ["sālā", "brother-in-law (wife's brother)"]
    assert cell(106,"right",1)[2] == "cīṛe"
    assert cell(106,"right",8)[2] == "gā̃"
    assert cell(106,"right",13)[2] == "măiṇā"
    assert cell(106,"right",33)[2] == "pălĕādū"
    assert cell(107,"left",2)[14].endswith(" verb")
    assert cell(107,"left",2,2)[14].endswith(" noun")
    assert cell(107,"left",10,3)[2] == "-o"
    assert cell(107,"left",10,3)[14].endswith(" suffix")
    assert cell(107,"right",28,3)[14].endswith(" relative")
    assert cell(108,"left",31,2)[14].endswith(" caus")
    assert cell(108,"right",25,4)[3] == "sister-in-law (husband's sister)"
    assert cell(109,"right",19,3)[14].endswith(" relative")
    assert cell(109,"right",23)[14].endswith(" instr")
    assert cell(109,"right",29)[2] == "tŭs dā"


def test_northern_hindko_mapping_and_source_lect():
    rows = installed()
    assert all(r[0] == "Northern Hindko" and r[14].startswith(source.DIALECT) for r in rows)
    assert all(r[11] == "" for r in rows)
    assert {a["source_lect"] for a in audited()} == {"Kāgānī of the Kāgān Valley"}
    with (DATA / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {r["ID"]: r for r in csv.DictReader(stream)}
    assert languages["Northern Hindko"]["Glottocode"] == "nort2662"
    assert languages["awan"]["Glottocode"] == "avan1234"
    with (DATA / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {r["ID"]: r for r in csv.DictReader(stream)}
    d = dialects["bailey1920-kagani"]
    assert d["Language_ID"] == "Northern Hindko" and d["Tag"] == source.DIALECT
    assert d["Latitude"] == d["Longitude"] == ""


def test_source_notes_and_profile_coverage():
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    import source_meta, profile_policy
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[2] and "�" not in row[2]
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        form = unicodedata.normalize("NFC", row[2])
        converted = unicodedata.normalize("NFC", tokenizer(form, column="IPA").replace(" ", "").replace("#", " "))
        assert converted == form.replace("w", "v").replace("ṅ", "ŋ")
        assert set(row[14].replace(source.DIALECT, "").split()) <= set(GRAMMATICAL_TAGS) | set(GENDER_TAGS)
    assert source_meta.SourceMeta().transcription("bailey1920kagani", CSV, "Northern Hindko")[0] == "bailey-kagani-1920"
    assert "bailey-kagani-1920" not in profile_policy.audit({})
    rows = {r[10]:r for r in installed()}
    assert "plain r" in rows["bailey1920kagani:p107:left:line:31"][6]
    assert "almost tsh" in rows["bailey1920kagani:p107:right:line:8"][6]
    assert "ruin" in rows["bailey1920kagani:p108:right:line:34"][6]


def test_scoped_parse_preserves_every_original():
    import make_cldf
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-kagani")
    raw = {r[10]:r for r in installed()}
    assert len(parsed) == stats["converted"] == 329 and not errors.getvalue()
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)


def test_all_source_see_also_references_survive():
    rows = {r[10]:r for r in installed()}
    for page, column, line, targets in [
        (107,"left",20,["one", "two"]),
        (107,"left",21,["right", "left"]),
        (107,"right",21,["pear"]),
        (108,"left",18,["medlar"]),
        (108,"right",2,["stand"]),
        (109,"right",8,["go", "come"]),
        (109,"right",17,["whither"]),
    ]:
        note = rows[f"bailey1920kagani:p{page}:{column}:line:{line}"][6]
        assert all(target in note for target in targets)
    excluded = {a["english_headword"]:a["printed_form_review"] for a in audited() if a["status"] == "exclude_crossref"}
    assert excluded == {"river":"[see stream]", "second":"[see two]", "spruce":"[see fir]"}


def test_fresh_independent_audit_pins_installed_output():
    import hashlib
    report = json.loads((PACKAGE / "independent-audit-20260926-pass2.json").read_text())
    assert report["sample_size"] == 20 and report["material_errors"] == 0
    assert report["result"] == "pass"
    # Reverse only the post-audit token-encoding fix; every lexical byte must
    # still match the immutable independently reviewed snapshot.
    audited_bytes = CSV.read_bytes().replace(b"dialect:Northern%20Hindko:", b"dialect:Northern Hindko:")
    assert hashlib.sha256(audited_bytes).hexdigest() == report["hashes"]["literal-staged.csv"]
    assert hashlib.sha256((PACKAGE / "audit.jsonl").read_bytes()).hexdigest() == report["hashes"]["literal-staged-audit.jsonl"]


def test_dialect_tag_is_one_registered_whitespace_free_token():
    from urllib.parse import unquote
    assert len(source.DIALECT.split()) == 1
    language = "Northern Hindko" if "kagani" in source.DIALECT else "Maria (India)"
    assert unquote(source.DIALECT.split(":")[1]) == language
    with (DATA / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        assert any(row["Tag"] == source.DIALECT and row["Language_ID"] == language
                   for row in csv.DictReader(stream))
