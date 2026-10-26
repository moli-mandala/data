"""Focused, source-local validation without a canonical install or full CLDF build."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata as ud
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "source_meta.py").exists())
sys.path.insert(0, str(DATA))
import make_cldf  # noqa: E402
import source_meta  # noqa: E402
from tags import GENDER_TAGS, GRAMMATICAL_TAGS  # noqa: E402
from pybtex.database import parse_file  # noqa: E402

STEM = "20260927-nair-adiya"
fields = ("lang", "param", "old_form", "gloss", "native", "ipa", "notes", "source",
          "cognateset", "etymology", "entry_key", "variant_of_key", "borrowed_from_key",
          "derivation_parent_keys", "tags")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    csv_path = HERE / "proposed.csv"
    audit_path = HERE / "proposal-audit.jsonl"
    first_path = HERE / "first-reading.jsonl"
    rows = list(csv.reader(csv_path.open(newline="")))
    first = [json.loads(s) for s in first_path.read_text().splitlines()]
    audit = [json.loads(s) for s in audit_path.read_text().splitlines()]
    assert len(rows) == 171 and all(len(r) == 15 for r in rows)
    assert len({r[10] for r in rows}) == 171
    assert len(first) == len(audit) == 29
    assert sum(len(x["witnesses"]) for x in first) == 174
    assert sum(x["held_or_excluded_count"] for x in audit) == 3
    assert {r[10] for r in rows} == {k for x in audit for k in x["entry_keys"]}
    assert {r[0] for r in rows} == {"Ravula", "Malayalam", "Kannada"}
    assert sum(r[0] == "Ravula" for r in rows) == 136
    by_key = {r[10]:r for r in rows}
    assert sum(bool(r[11]) for r in rows) == 12
    for r in rows:
        if r[11]:
            assert r[11] in by_key and r[11] != r[10]
            assert r[0] == by_key[r[11]][0]
    assert sum(r[0] in {"Malayalam", "Kannada"} for r in rows) == 35
    assert all(r[4] == r[5] == "" and r[7].startswith("nair2013adiya[") for r in rows)
    assert all(not any(ud.category(c) == "Co" or c == "\ufffd" for c in r[2]) for r in rows)
    for r in rows:
        assert all(t in GENDER_TAGS or t in GRAMMATICAL_TAGS or t.startswith("dialect:")
                   for t in r[14].split()),r
        assert ud.is_normalized("NFC",r[2])
    first_forms = [w["form"] for x in first for w in x["witnesses"]]
    assert {"kal̲uttŭ", "eel̲u", "aal̲am", "n̲aanu", "n̲aanka", "n̲aamu",
            "n̲inka", "n̲aaŋga barivoom", "nii eppoo ban̲ṛe?",
            "itŭ el̲uttŭ", "enaakku par̲r̲aa", "enaakku muun̲ru puḷḷe uḷa",
            "niinŋka", "niiŋka"} <= set(first_forms)
    bib_path = HERE / "metadata/proposed-references.bib"
    bib = parse_file(bib_path)
    assert set(bib.entries) == {"nair2013adiya", "sankarankutty1978adiya",
                                "syama1978adiya", "sarala1978adiya", "davidson1978adiya"}
    with tempfile.TemporaryDirectory(dir=HERE / "metadata") as temp:
        yaml_path = Path(temp) / f"{STEM}.yaml"
        yaml_path.write_bytes((HERE / "metadata/proposed-settings.yaml").read_bytes())
        meta = source_meta.SourceMeta(paths=[yaml_path])
        for language in ("Ravula", "Malayalam", "Kannada"):
            assert meta.transcription("nair2013adiya",f"{STEM}.csv",language)==(None,False)
            assert meta.transcription_rule("nair2013adiya",f"{STEM}.csv",language)["preserve_hyphens"]
        assert meta.file_flag(f"{STEM}.csv","identity","dedupe_by_entry_key") is True
        with patch.object(make_cldf.source_meta,"load",return_value=meta):
            errors=io.StringIO()
            parsed,stats=make_cldf.parse_file(str(csv_path),errors,name=STEM)
        assert errors.getvalue()=="",errors.getvalue()
        assert len(parsed)==len(rows) and stats=={"converted":0,"for_conversion":0},stats
        for parsed_row,raw in zip(parsed,rows):
            assert [getattr(parsed_row,f) for f in fields]==raw,raw[10]
            assert parsed_row.form==raw[2]
    report={"status":"pass: focused actual parser and accounting; independent review pending",
            "csv_sha256":sha(csv_path),"audit_sha256":sha(audit_path),
            "first_reading_sha256":sha(first_path),"settings_sha256":sha(HERE/"metadata/proposed-settings.yaml"),
            "bib_sha256":sha(bib_path),"parsed_rows":len(rows),"physical_witnesses":174,
            "source_books_reported":4,"canonical_changes":False}
    (HERE/"candidate-validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(report["status"])

if __name__=="__main__":
    main()
