"""Focused checks for Bailey 1920 complete Rampur glossary."""

import csv
import importlib.util
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/bailey_rampur_1920"
CSV = DATA / "data/other/forms/20260925-bailey-rampur.csv"
PROFILE = DATA / "conversion/bailey-rampur-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_rampur_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_first_page_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 245 and len(rows) == 288
    assert Counter(a["status"] for a in audit) == {
        "ingested": 242, "hold_typography": 1, "cross_reference": 2,
    }
    assert [(a["printed_page"],a["vocabulary_item"]) for a in audit] == [(p,i) for p,n in [(144,59),(145,66),(146,63),(147,57)] for i in range(1,n+1)]
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(p,p+26) for p in range(144,148)}
    assert all(a["canonical_language"] == "ramp" for a in audit)
    assert len({r[10] for r in rows}) == len(rows)


def test_controls_semantic_holds_and_seeded_image_review():
    audit = {a["vocabulary_item"]: a for a in audited()[:59]}
    rows = {r[10]: r for r in installed()}
    assert audit[1]["status"] == audit[35]["status"] == "cross_reference"
    assert all(audit[i]["status"] == "ingested" for i in (15, 30, 32, 46, 48, 50))
    assert audit[19]["status"] == "hold_typography" and "stacked" in audit[19]["reason"]
    assert rows["bailey1920rampur:p144:item:6"][2:4] == ["patsha", "backwards"]
    assert rows["bailey1920rampur:p144:item:14"][2:4] == ["patsha", "behind"]
    assert all(r[11] == "" for r in rows.values())
    with (PACKAGE / "transcription.tsv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    with (PACKAGE / "sample-review-20.tsv").open(encoding="utf-8", newline="") as stream:
        reviewed = list(csv.DictReader(stream, delimiter="\t"))
    sampled = random.Random(1920).sample(source_rows, 20)
    assert [int(r["item"]) for r in reviewed] == [int(r["item"]) for r in sampled]
    assert all(r["source_cell"] == s["rampur_form_review"] and r["decision"] == s["decision"]
               for r, s in zip(reviewed, sampled))
    assert all(r["visual_check"] == "pass" and r["material_error"] == "0" for r in reviewed)


def test_metadata_profile_and_parse():
    import make_cldf
    import profile_policy
    import source_meta

    assert "@book{bailey1920rampur," in (DATA / "cldf/sources.bib").read_text()
    assert "ramp" in {r["ID"] for r in csv.DictReader((DATA / "cldf/languages.csv").open())}
    assert source_meta.SourceMeta().transcription("bailey1920rampur", CSV, "ramp")[0] == "bailey-rampur-1920"
    assert "bailey-rampur-1920" not in profile_policy.audit({})
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "ramp"
        assert row[7] == f"bailey1920rampur[p. {row[10].split(':')[1][1:]}, vocabulary item {row[10].split(':')[3]}, Rampur column]"
        assert row[1] == row[4] == row[5] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        assert "�" not in tokenizer(row[2], column="IPA")
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-rampur")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 288
    raw = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)
    assert next(r for r in parsed if r.entry_key.endswith(":22")).form == "rōṭṭi"


def test_literal_marks_and_subanswer_structure():
    rows={r[10]:r for r in installed()}
    assert rows["bailey1920rampur:p145:item:19"][2] == "năs̲h̲ṇo"
    assert rows["bailey1920rampur:p144:item:15:answer2"][14] == "prep"
    assert rows["bailey1920rampur:p145:item:20:answer2"][2] == "băkri"
    assert rows["bailey1920rampur:p145:item:20:answer2"][14] == "f"
    assert rows["bailey1920rampur:p145:item:66:answer5"][3] == "as much (or many, relative)"
    tokenizer=Tokenizer(str(PROFILE))
    assert tokenizer("s̲h̲ūṇṇo",column="IPA").replace(" ","") == "s̲h̲ūṇṇo"
    assert tokenizer("wār",column="IPA").replace(" ","") == "vār"


def test_independently_corrected_literal_distinctions():
    rows = {r[10]: r for r in installed()}
    expected = {
        (144, 11, 2): "tsīkṇo",
        (144, 15, 1): "āhndi",
        (144, 44, 1): "āhndi",
        (145, 32, 1): "ḍaũk",
        (145, 49, 1): "s̲h̲īkṇo",
        (145, 65, 1): "ḍaũk",
        (146, 29, 1): "ŭdzu khăṛno",
        (147, 25, 3): "dujjau",
        (147, 29, 1): "āhndī",
        (147, 31, 1): "bāro",
        (147, 41, 1): "gīũh",
    }
    for (page, item, answer), form in expected.items():
        key = f"bailey1920rampur:p{page}:item:{item}"
        if answer > 1:
            key += f":answer{answer}"
        assert rows[key][2] == form
    from tags import GENDER_TAGS, GRAMMATICAL_TAGS
    assert {tag for row in rows.values() for tag in row[14].split()} <= GENDER_TAGS | GRAMMATICAL_TAGS


def test_fresh_independent_full_scope_audit():
    import hashlib
    report = json.loads((PACKAGE / 'independent-full-scope-audit-20260926-pass4.json').read_text())
    assert report['sample_size'] == 20 and report['material_errors'] == 0
    source_rows = {(r['page'], r['item']): r for r in source.read_source()}
    for entry in report['entries']:
        row = source_rows[(entry['page'], entry['item'])]
        assert row['rampur_form_review'] == entry['rampur_form_review']
        assert row['gloss'] == entry['gloss']
    final = json.loads((PACKAGE / 'grammar-completion-20260926.json').read_text())
    for name, digest in final['input_sha256'].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == final['installed_sha256']
