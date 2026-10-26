"""Bounded actual SourceMeta, tags, bibliography formatter and Tokenizer validation."""
import csv
import json
from pathlib import Path
import sys
import unicodedata as ud

import prepare_proposal as proposal

HERE = proposal.HERE
ROOT = proposal.ROOT
sys.path.insert(0, str(ROOT / "data"))
import source_meta
from dialects import dialect_tag, normalize_dialect
from tags import GRAMMATICAL_TAGS, GENDER_TAGS
from segments.tokenizer import Tokenizer
from pybtex.database import parse_file
from pybtex import PybtexEngine


def validate():
    status = json.loads((HERE / "proposal-status.json").read_text())
    for path, digest in status["input_hashes"].items():
        assert proposal.sha(ROOT / path) == digest, path
    for path, digest in status["output_hashes"].items():
        assert proposal.sha(HERE / path) == digest, path
    settings = HERE / f"{proposal.STEM}.yaml"
    meta = source_meta.SourceMeta([*source_meta.yaml_files(), settings])
    ref = meta.source("grierson1908ahirwati")["reference"]
    assert ref == dict(editor="OpenAI Codex", ocr=False, etymology_provenance="source")
    for language in ("Ahirwadi", "mewati_jhambaus", "Bagri", "bagri_fatehabad", "Haryanvi", "kaithal", "Mewati", "mewati_akera"):
        assert meta.transcription("grierson1908ahirwati", f"{proposal.STEM}.csv", language) == (proposal.PROFILE, True)
        assert meta.transcription_rule("grierson1908ahirwati", f"{proposal.STEM}.csv", language)["preserve_hyphens"] is True
    languages = {r["ID"]: r for r in csv.DictReader((ROOT / "data/cldf/languages.csv").open())}
    existing = {r["ID"] for r in csv.DictReader((ROOT / "data/cldf/dialects.csv").open())}
    lects = list(csv.DictReader((HERE / "dialects-proposed.csv").open()))
    assert len(lects) == 5
    for row in lects:
        assert row["ID"] not in existing
        assert row["Language_ID"] in languages
        assert not any(row[field] for field in ("Glottocode", "Latitude", "Longitude"))
        assert row["Tag"] == dialect_tag(row["Language_ID"], row["Source_Language_ID"], row["Name"])
        assert len(row["Tag"].split()) == 1
        assert normalize_dialect(row["Source_Language_ID"], row["Tag"], {}, set(languages)) == (row["Language_ID"], row["Tag"])
    bib = parse_file(str(HERE / "references-proposed.bib"))
    current_bib = parse_file(str(ROOT / "data/cldf/sources.bib"))
    assert set(bib.entries) == {"grierson1908ahirwati"}
    assert not set(bib.entries) & set(current_bib.entries)
    formatted = PybtexEngine().format_from_string(bib.to_string("bibtex"), "plain", output_backend="markdown")
    assert "1908" in formatted and "Rajasthani" in formatted and "Gujarati" in formatted
    tokenizer = Tokenizer(str(HERE / f"{proposal.PROFILE}.txt"))
    deltas, cases, tag_errors = [], 0, []
    for row in proposal.records(HERE / "coverage-corpus.jsonl"):
        for token in row["tags"]:
            if not token.startswith("dialect:") and token not in GRAMMATICAL_TAGS | GENDER_TAGS:
                tag_errors.append(dict(key=row["key"], tag=token))
        for norm in ("NFC", "NFD"):
            actual = ud.normalize("NFC", tokenizer(ud.normalize(norm, row["original"]), column="IPA").replace(" ", "").replace("#", " "))
            assert actual == row["expected_display"], (row["key"], norm, actual, row["expected_display"])
            assert "�" not in actual
            assert not any(ud.category(c) == "Co" for c in actual)
            cases += 1
        if row["original"] != row["expected_display"]:
            deltas.append(dict(key=row["key"], original=row["original"], display=row["expected_display"], reason="NFC/lowercase/ṅ→ŋ only"))
    assert not tag_errors, tag_errors
    with (HERE / "display-deltas.jsonl").open("w") as stream:
        for row in deltas:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(status="PASS preliminary isolated real APIs; no canonical activation or final source acceptance", source_metadata="Actual SourceMeta with all canonical YAMLs plus one isolated proposal; reference key declared once", reference_flags=ref, bibliography_keys=list(bib.entries), bibliography_formatted=formatted,
                  dialect_count=len(lects), dialects=lects, tokenizer_normalization_cases=cases, fixture_cases=status["fixture_cases"], display_delta_rows=len(deltas), unregistered_tags=tag_errors,
                  input_hashes=status["input_hashes"], config_hashes=status["output_hashes"], display_deltas_sha256=proposal.sha(HERE / "display-deltas.jsonl"),
                  caveats=["Production SourceMeta and segments Tokenizer APIs used without parser monkeypatches. No final combined parse_file claim: final CSV is not yet frozen.", "Profile stage is source-local only; canonical profile registration and installed-route parser are later gates.", "Original is copied exactly to fixture corpus and never converted; conversion affects separate display only.", "Native evidence remains unchanged; source-native alignment qualifications are not resolved by this conversion test.", *status["pending"]], canonical_changes=False)
    (HERE / "actual-api-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ("status", "dialect_count", "fixture_cases", "tokenizer_normalization_cases", "display_delta_rows", "unregistered_tags")}, ensure_ascii=False))


if __name__ == "__main__":
    validate()
